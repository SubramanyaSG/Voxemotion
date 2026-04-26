"""
app.py – VoxEmotion TTS v4 (Auth + Security)
Firebase Firestore for user profiles | bcrypt passwords
CSRF protection | Rate limiting | Secure session cookies
File uploads processed locally – never stored in cloud
"""
import os, sys, json, re, uuid, traceback, time, hmac, hashlib, secrets
from datetime import datetime, timedelta, timezone
from functools import wraps
from pathlib import Path
import numpy as np
import soundfile as sf
import scipy.interpolate as interp
from flask import (Flask, request, jsonify, render_template, Response,
                   redirect, url_for, session, flash, abort)
from flask_cors import CORS
import warnings
warnings.filterwarnings('ignore')

# ── soxr / numpy resampler ────────────────────────────────────────────────────
try:
    import soxr
    def _resample(data, sr_in, sr_out):
        return soxr.resample(data.astype(np.float32), sr_in, sr_out, quality='HQ')
except ImportError:
    def _resample(data, sr_in, sr_out):
        if sr_in == sr_out: return data
        n = int(len(data) * sr_out / sr_in)
        return np.interp(np.linspace(0,len(data)-1,n),
                         np.arange(len(data)), data).astype(np.float32)

# ── Runtime config ────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
_CFG_PATH = os.path.join(BASE_DIR, 'runtime_config.json')
_CFG = {}
if os.path.exists(_CFG_PATH):
    with open(_CFG_PATH) as f: _CFG = json.load(f)

DATASET_ROOT = os.environ.get('DATASET_ROOT', _CFG.get('DATASET_ROOT',''))
OUTPUT_DIR   = os.environ.get('OUTPUT_DIR',   _CFG.get('OUTPUT_DIR',  'outputs'))
MODEL_DIR    = os.environ.get('MODEL_DIR',    _CFG.get('MODEL_DIR',   'models'))
SAMPLE_RATE  = int(_CFG.get('SAMPLE_RATE', 22050))
EMOTIONS     = _CFG.get('EMOTIONS', ['angry','happy','neutral','sad','surprise'])
SECRET_KEY   = os.environ.get('FLASK_SECRET_KEY', secrets.token_hex(32))

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)

# ════════════════════════════════════════════════════════════════════════════
# FIREBASE SETUP
# Free tier: Firestore (1 GB storage, 50k reads/day, 20k writes/day)
# Setup steps:
#   1. https://console.firebase.google.com → New project "voxemotion"
#   2. Project Settings → Service Accounts → Generate private key
#   3. Save as firebase_credentials.json in project root
#   4. Firestore Database → Create database (test mode)
#   5. Authentication → Sign-in method → Email/Password → Enable
# ════════════════════════════════════════════════════════════════════════════
_fb_app = _fb_db = _fb_auth_mod = None
_USE_FIREBASE = False

def init_firebase():
    global _fb_app, _fb_db, _fb_auth_mod, _USE_FIREBASE
    creds_path = os.path.join(BASE_DIR, 'firebase_credentials.json')
    if not os.path.exists(creds_path):
        print('⚠  firebase_credentials.json not found → LOCAL mode (users.json)')
        return False
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore, auth as fb_auth
        if not firebase_admin._apps:
            cred = credentials.Certificate(creds_path)
            _fb_app = firebase_admin.initialize_app(cred)
        else:
            _fb_app = firebase_admin.get_app()
        _fb_db         = firestore.client()
        _fb_auth_mod   = fb_auth
        _USE_FIREBASE  = True
        print('✅  Firebase Firestore connected.')
        return True
    except Exception as e:
        print(f'⚠  Firebase init failed: {e} → LOCAL mode')
        return False

init_firebase()

# ── Local user store fallback ─────────────────────────────────────────────────
_LOCAL_USERS = os.path.join(BASE_DIR, 'users.json')
def _lu(): return json.load(open(_LOCAL_USERS)) if os.path.exists(_LOCAL_USERS) else {}
def _su(u): json.dump(u, open(_LOCAL_USERS,'w'), indent=2)

# ── Password hashing ──────────────────────────────────────────────────────────
try:
    import bcrypt
    def hash_pw(pw):  return bcrypt.hashpw(pw.encode(), bcrypt.gensalt(12)).decode()
    def check_pw(pw, h): return bcrypt.checkpw(pw.encode(), h.encode())
except ImportError:
    def hash_pw(pw):
        salt = os.urandom(32).hex()
        h    = hashlib.pbkdf2_hmac('sha256', pw.encode(), salt.encode(), 310000)
        return f"{salt}${h.hex()}"
    def check_pw(pw, hashed):
        try:
            salt, h = hashed.split('$',1)
            c = hashlib.pbkdf2_hmac('sha256', pw.encode(), salt.encode(), 310000)
            return hmac.compare_digest(c.hex(), h)
        except: return False

# ── CSRF ──────────────────────────────────────────────────────────────────────
def gen_csrf():
    t = secrets.token_hex(32); session['csrf_token'] = t; return t
def ok_csrf(t):
    s = session.get('csrf_token')
    return bool(s and t and hmac.compare_digest(s, t))

# ── Rate limiter (in-memory) ──────────────────────────────────────────────────
_rl: dict = {}
def rate_ok(ip, limit, window):
    now = time.time()
    calls = [x for x in _rl.get(ip,[]) if now-x < window]
    if len(calls) >= limit: return False
    calls.append(now); _rl[ip] = calls; return True

# ── Registration cooldown ─────────────────────────────────────────────────────
_reg_cd: dict = {}
def reg_wait(ip): return max(0, int(30-(time.time()-_reg_cd.get(ip,0))))
def set_reg_cd(ip): _reg_cd[ip] = time.time()

# ── Password reset tokens ─────────────────────────────────────────────────────
_rst: dict = {}
def make_rst(email):
    t = secrets.token_urlsafe(48)
    _rst[t] = {'email':email,'exp':time.time()+900,'used':False}; return t
def use_rst(token):
    r = _rst.get(token)
    if not r or r['used'] or time.time()>r['exp']: return None
    r['used']=True; return r['email']

# ── Email sending ─────────────────────────────────────────────────────────────
import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

SMTP_EMAIL    = os.environ.get('SMTP_EMAIL', 'support_voxemotion@gmail.com')
SMTP_PASS     = os.environ.get('SMTP_APP_PASSWORD', '')
APP_URL       = os.environ.get('APP_BASE_URL', 'http://127.0.0.1:5000')

def send_email(to, subject, html):
    if not SMTP_PASS:
        print(f'[EMAIL] To:{to} | Subject:{subject}')
        print(f'[EMAIL] (No SMTP configured – showing in console)')
        return True
    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject; msg['From'] = f'VoxEmotion <{SMTP_EMAIL}>'; msg['To'] = to
        msg.attach(MIMEText(html,'html'))
        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL('smtp.gmail.com',465,context=ctx) as s:
            s.login(SMTP_EMAIL, SMTP_PASS); s.sendmail(SMTP_EMAIL, to, msg.as_string())
        return True
    except Exception as e:
        print(f'Email error: {e}'); return False

def send_reset_email(email, token):
    link = f'{APP_URL}/reset-password/{token}'
    html = f"""<div style="font-family:sans-serif;background:#0d0d1a;color:#e8e8f8;padding:2rem;border-radius:12px;max-width:480px">
    <h2 style="color:#6e56ff">VoxEmotion – Reset Password</h2>
    <p>Click below to set a new password. <b>Link expires in 15 minutes.</b></p>
    <a href="{link}" style="display:inline-block;margin:1rem 0;padding:.8rem 1.5rem;
       background:#6e56ff;color:#fff;border-radius:8px;text-decoration:none;font-weight:600">
      Reset My Password</a>
    <p style="color:#7878a8;font-size:.8rem">If you didn't request this, ignore this email.</p>
    <p style="color:#7878a8;font-size:.8rem">Support: support_voxemotion@gmail.com</p></div>"""
    send_email(email, 'VoxEmotion – Reset Your Password', html)

# ── User CRUD ─────────────────────────────────────────────────────────────────
def user_exists(email):
    if _USE_FIREBASE:
        try: _fb_auth_mod.get_user_by_email(email); return True
        except: return False
    return email in _lu()

def create_user(email, password, fullname, dob):
    h = hash_pw(password)
    if _USE_FIREBASE:
        try:
            u = _fb_auth_mod.create_user(email=email, display_name=fullname)
            _fb_db.collection('users').document(u.uid).set({
                'fullname':fullname,'dob':dob,'email':email,'pw_hash':h,
                'created':datetime.now(timezone.utc).isoformat(),'role':'user'})
            return True
        except Exception as e: print(f'Firebase create_user: {e}'); return False
    else:
        u = _lu()
        u[email] = {'fullname':fullname,'dob':dob,'pw_hash':h,
                    'role':'user','created':datetime.now().isoformat()}
        _su(u); return True

def verify_user(email, password):
    if _USE_FIREBASE:
        try:
            fu = _fb_auth_mod.get_user_by_email(email)
            d  = _fb_db.collection('users').document(fu.uid).get()
            if d.exists:
                data = d.to_dict()
                if check_pw(password, data.get('pw_hash','')):
                    return {'email':email,'fullname':data.get('fullname',''),
                            'uid':fu.uid,'role':data.get('role','user')}
        except: return None
    else:
        u = _lu(); r = u.get(email)
        if r and check_pw(password, r.get('pw_hash','')):
            return {'email':email,'fullname':r.get('fullname',''),'role':r.get('role','user')}
    return None

def update_password(email, new_pw):
    h = hash_pw(new_pw)
    if _USE_FIREBASE:
        try:
            fu = _fb_auth_mod.get_user_by_email(email)
            _fb_auth_mod.update_user(fu.uid, password=new_pw)
            _fb_db.collection('users').document(fu.uid).update({'pw_hash':h})
            return True
        except Exception as e: print(f'Firebase update_pw: {e}'); return False
    else:
        u = _lu()
        if email in u: u[email]['pw_hash']=h; _su(u); return True
        return False

# ── Validators ────────────────────────────────────────────────────────────────
def vld_email(e): return bool(re.match(r'^[^\s@]+@[^\s@]+\.[^\s@]{2,}$', e))
def vld_pw(p):    return len(p)>=8 and bool(re.search(r'[A-Z]',p)) and bool(re.search(r'[0-9]',p))
def vld_name(n):  return bool(re.match(r'^[A-Za-z]+$', n))
def vld_dob(d):
    try:
        if not re.match(r'^\d{2}/\d{2}/\d{4}$',d): return False
        dd,mm,yy = map(int,d.split('/'))
        age = (datetime.now()-datetime(yy,mm,dd)).days/365.25
        return 5 <= age <= 120
    except: return False

def login_required(f):
    @wraps(f)
    def w(*a,**k):
        if 'user_email' not in session:
            flash('Please log in to continue.','info')
            return redirect(url_for('login_page'))
        return f(*a,**k)
    return w

# ════════════════════════════════════════════════════════════════════════════
# AUDIO PIPELINE
# ════════════════════════════════════════════════════════════════════════════
T2_MAX_CHARS    = 150
SILENCE_SAMPLES = int(SAMPLE_RATE * 0.18)
EMOTION_PARAMS  = {'angry':(2.0,1.30,1.05),'happy':(3.5,1.20,1.10),
                   'neutral':(0.0,1.00,1.00),'sad':(-3.0,0.75,0.88),
                   'surprise':(4.5,1.15,1.15)}

def load_audio(path, tsr=SAMPLE_RATE, maxs=None):
    d,sr = sf.read(path,dtype='float32',always_2d=False)
    if d.ndim>1: d=d.mean(axis=1)
    if maxs: d=d[:int(sr*maxs)]
    if sr!=tsr: d=_resample(d,sr,tsr)
    return d.astype(np.float32),tsr

def save_audio(path, audio, sr=SAMPLE_RATE):
    sf.write(path,np.clip(audio,-1,1).astype(np.float32),sr)

def _pitch(a,sr,n):
    if n==0: return a
    t=max(1,int(len(a)/2.0**(n/12.0)))
    s=np.interp(np.linspace(0,len(a)-1,t),np.arange(len(a)),a)
    return np.interp(np.linspace(0,len(s)-1,len(a)),np.arange(len(s)),s).astype(np.float32)

def _stretch(a,r):
    if r==1: return a
    t=max(1,int(len(a)/r))
    return np.interp(np.linspace(0,len(a)-1,t),np.arange(len(a)),a).astype(np.float32)

def apply_emo(audio, sr, emo):
    ps,es,ts=EMOTION_PARAMS.get(emo,(0,1,1))
    if ps: audio=_pitch(audio,sr,ps)
    audio=audio*es
    if ts!=1: audio=_stretch(audio,ts)
    return np.clip(audio,-1,1).astype(np.float32)

try:
    import inflect; from unidecode import unidecode
    _inf=inflect.engine()
    def norm(t):
        t=unidecode(str(t)); t=re.sub(r'\d+',lambda m:_inf.number_to_words(m.group()),t)
        t=re.sub(r"[^a-zA-Z0-9\s.,!?'\-]",'',t); return re.sub(r'\s+',' ',t).strip()
except ImportError:
    def norm(t):
        t=re.sub(r"[^a-zA-Z0-9\s.,!?'\-]",'',str(t)); return re.sub(r'\s+',' ',t).strip()

def chunks(text, mx=T2_MAX_CHARS):
    raw=re.split(r'(?<=[.!?])\s+',text.strip()); out,buf=[],''
    for s in raw:
        s=s.strip()
        if not s: continue
        if len(buf)+len(s)+1<=mx: buf=(buf+' '+s).strip() if buf else s
        else:
            if buf: out.append(buf)
            if len(s)>mx:
                ps=re.split(r'(?<=,)\s+',s); sub=''
                for p in ps:
                    if len(sub)+len(p)+1<=mx: sub=(sub+' '+p).strip() if sub else p
                    else:
                        if sub: out.append(sub)
                        while len(p)>mx: out.append(p[:mx]); p=p[mx:]
                        sub=p
                if sub: out.append(sub)
            else: buf=s
    if buf: out.append(buf)
    return [c for c in out if c.strip()]

def extract_pdf(p):
    try:
        from pdfminer.high_level import extract_text; return extract_text(p)
    except:
        try:
            import PyPDF2
            with open(p,'rb') as f:
                r=PyPDF2.PdfReader(f); return '\n'.join(pg.extract_text() or '' for pg in r.pages)
        except Exception as e: return f'[PDF:{e}]'

def extract_docx(p):
    try:
        from docx import Document; return '\n'.join(pg.text for pg in Document(p).paragraphs)
    except Exception as e: return f'[DOCX:{e}]'

def extract_txt(p):
    try:
        with open(p,'r',encoding='utf-8',errors='replace') as f: return f.read()
    except Exception as e: return f'[TXT:{e}]'

import pandas as pd
_dfc=None
def get_df():
    global _dfc
    if _dfc is not None: return _dfc
    csv=os.path.join(OUTPUT_DIR,'dataset_metadata.csv')
    if os.path.exists(csv): _dfc=pd.read_csv(csv); return _dfc
    rec=[]
    if DATASET_ROOT and os.path.isdir(DATASET_ROOT):
        for spk in sorted(Path(DATASET_ROOT).iterdir()):
            if not spk.is_dir(): continue
            for emo in sorted(d for d in spk.iterdir() if d.is_dir()):
                for wav in sorted(emo.glob('*.wav')):
                    try:
                        i=sf.info(str(wav))
                        rec.append({'speaker':spk.name,'emotion':emo.name.lower(),
                                    'file':str(wav),'filename':wav.name,
                                    'text':'','duration':i.duration,'readable':True})
                    except: pass
    _dfc=pd.DataFrame(rec); return _dfc

_t2=_wg=_ut=None; _t2f=False
def _load_t2():
    global _t2,_wg,_ut,_t2f
    if _t2f or _t2: return _t2 is not None
    try:
        import torch
        dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _t2=torch.hub.load('NVIDIA/DeepLearningExamples:torchhub','nvidia_tacotron2',
                            model_math='fp32',verbose=False).to(dev).eval()
        _wg=torch.hub.load('NVIDIA/DeepLearningExamples:torchhub','nvidia_waveglow',
                            model_math='fp32',verbose=False).to(dev).eval()
        _ut=torch.hub.load('NVIDIA/DeepLearningExamples:torchhub','nvidia_tts_utils',verbose=False)
        for m in _wg.modules():
            if hasattr(m,'weight_v'):
                import torch.nn.utils as nu; nu.remove_weight_norm(m)
        return True
    except Exception as e:
        print(f'Tacotron2 unavailable: {e}'); _t2f=True; return False

def _t2c(chunk):
    import torch; dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    s,l=_ut.prepare_input_sequence([chunk])
    with torch.no_grad():
        mel,_,_=_t2.infer(s.to(dev),l.to(dev)); a=_wg.infer(mel)
    return a[0].cpu().numpy().astype(np.float32)

def _ret(emo, dur=3.0):
    df=get_df(); ok='readable' in df.columns
    sub=(df[(df['readable']==True)&(df['emotion']==emo)] if ok
         else df[df['emotion']==emo])
    if sub.empty: sub=df[df['readable']==True] if ok else df
    if sub.empty: return np.zeros(int(SAMPLE_RATE*dur),dtype=np.float32)
    try:
        a,_=load_audio(sub.sample(1).iloc[0]['file'],SAMPLE_RATE,maxs=dur)
        n=int(SAMPLE_RATE*dur)
        if len(a)<n: a=np.pad(a,(0,n-len(a)))
        return a
    except: return np.zeros(int(SAMPLE_RATE*dur),dtype=np.float32)

def synthesize_audio(text, emotion):
    emotion=emotion.lower() if emotion.lower() in EMOTION_PARAMS else 'neutral'
    text=norm(text)[:10_000_000]
    cks=chunks(text) or [text[:T2_MAX_CHARS]]
    use_t2=_load_t2()
    silence=np.zeros(SILENCE_SAMPLES,dtype=np.float32)
    parts=[]; sr=22050 if use_t2 else SAMPLE_RATE
    for i,ck in enumerate(cks):
        seg=None
        if use_t2:
            try: seg=_t2c(ck)
            except Exception as e: print(f'Chunk {i+1}: {e}')
        if seg is None:
            seg=_ret(emotion, max(2.0,len(ck.split())/2.5)); sr=SAMPLE_RATE
        parts.append(seg)
        if i<len(cks)-1: parts.append(silence)
    audio=np.concatenate(parts).astype(np.float32) if parts else np.zeros(sr*3,dtype=np.float32)
    audio=apply_emo(audio,sr,emotion)
    fname=f'tts_{emotion}_{uuid.uuid4().hex[:8]}.wav'
    save_audio(os.path.join(OUTPUT_DIR,fname),audio,sr)
    return {'file':fname,'text':text,'emotion':emotion,'duration':round(len(audio)/sr,2)}

# ════════════════════════════════════════════════════════════════════════════
# FLASK APP
# ════════════════════════════════════════════════════════════════════════════
app = Flask(__name__,
            template_folder=os.path.join(BASE_DIR,'templates'),
            static_folder=os.path.join(BASE_DIR,'static'))
app.secret_key = SECRET_KEY
app.config.update(
    SESSION_COOKIE_HTTPONLY  = True,
    SESSION_COOKIE_SAMESITE  = 'Lax',
    SESSION_COOKIE_SECURE    = False,  # True in production HTTPS
    PERMANENT_SESSION_LIFETIME = timedelta(days=7),
    MAX_CONTENT_LENGTH       = 50*1024*1024,
)
CORS(app, supports_credentials=True)

@app.after_request
def security_headers(resp):
    resp.headers['X-Content-Type-Options']  = 'nosniff'
    resp.headers['X-Frame-Options']         = 'DENY'
    resp.headers['X-XSS-Protection']        = '1; mode=block'
    resp.headers['Referrer-Policy']         = 'strict-origin-when-cross-origin'
    resp.headers['Permissions-Policy']      = 'microphone=(), camera=()'
    resp.headers['Content-Security-Policy'] = (
        "default-src 'self'; script-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://fonts.gstatic.com; "
        "font-src 'self' https://fonts.gstatic.com; connect-src 'self'; "
        "img-src 'self' data:; media-src 'self' blob:;")
    return resp

# ── Auth routes ───────────────────────────────────────────────────────────────
@app.route('/login', methods=['GET','POST'])
def login_page():
    if 'user_email' in session: return redirect(url_for('index'))
    csrf = gen_csrf()
    if request.method == 'POST':
        if not ok_csrf(request.form.get('csrf_token','')):
            flash('Invalid request. Please try again.','error'); return redirect(url_for('login_page'))
        ip = request.remote_addr
        if not rate_ok(ip,10,60):
            flash('Too many attempts. Wait 1 minute.','error')
            return render_template('login.html', csrf_token=csrf)
        email = request.form.get('email','').strip().lower()
        pw    = request.form.get('password','')
        if not vld_email(email) or not pw:
            flash('Enter a valid email and password.','error')
            return render_template('login.html', csrf_token=csrf)
        user = verify_user(email, pw)
        if not user:
            flash('Incorrect email or password.','error')
            return render_template('login.html', csrf_token=csrf)
        session.permanent = bool(request.form.get('remember'))
        session['user_email']    = user['email']
        session['user_fullname'] = user.get('fullname','')
        session['user_role']     = user.get('role','user')
        session.pop('csrf_token',None)
        return redirect(url_for('index'))
    return render_template('login.html', csrf_token=csrf)

@app.route('/register', methods=['POST'])
def register():
    csrf = gen_csrf()
    if not ok_csrf(request.form.get('csrf_token','')):
        flash('Invalid request.','error'); return redirect(url_for('login_page'))
    ip = request.remote_addr
    w  = reg_wait(ip)
    if w:
        flash(f'Wait {w}s before registering again.','warning')
        return render_template('login.html', csrf_token=csrf)
    if not rate_ok(ip,3,3600):
        flash('Registration limit reached. Try later.','error')
        return render_template('login.html', csrf_token=csrf)
    fn   = request.form.get('fullname','').strip()
    dob  = request.form.get('dob','').strip()
    em   = request.form.get('email','').strip().lower()
    pw   = request.form.get('password','')
    cpw  = request.form.get('confirm_password','')
    errs = []
    if not vld_name(fn):  errs.append('Name must contain letters only — no spaces or special characters.')
    if not vld_dob(dob):  errs.append('Invalid date. Use DD/MM/YYYY.')
    if not vld_email(em): errs.append('Invalid email address.')
    if not vld_pw(pw):    errs.append('Password: 8+ chars, 1 uppercase, 1 number.')
    if pw != cpw:         errs.append('Passwords do not match.')
    if not request.form.get('agree_terms'): errs.append('Accept the Terms & Conditions.')
    if errs:
        for e in errs: flash(e,'error')
        return render_template('login.html', csrf_token=csrf)
    if user_exists(em):
        flash('Email already registered.','error')
        return render_template('login.html', csrf_token=csrf)
    if not create_user(em, pw, fn, dob):
        flash('Registration failed. Try again.','error')
        return render_template('login.html', csrf_token=csrf)
    set_reg_cd(ip)
    flash('Account created! Please log in.','success')
    return redirect(url_for('login_page'))

@app.route('/forgot-password', methods=['POST'])
def forgot_password():
    csrf = gen_csrf()
    if not ok_csrf(request.form.get('csrf_token','')):
        flash('Invalid request.','error'); return redirect(url_for('login_page'))
    if not rate_ok(request.remote_addr,3,300):
        flash('Too many reset requests. Wait 5 minutes.','error')
        return render_template('login.html', csrf_token=csrf)
    em = request.form.get('email','').strip().lower()
    if not vld_email(em):
        flash('Enter a valid email.','error')
        return render_template('login.html', csrf_token=csrf)
    if user_exists(em):          # Always same response to prevent enumeration
        send_reset_email(em, make_rst(em))
    flash('If that email is registered, a reset link has been sent.','success')
    return redirect(url_for('login_page'))

@app.route('/reset-password/<token>', methods=['GET','POST'])
def reset_password(token):
    csrf = gen_csrf()
    if request.method=='GET':
        r = _rst.get(token)
        if not r or r['used'] or time.time()>r['exp']:
            flash('Reset link invalid or expired.','error'); return redirect(url_for('login_page'))
        return render_template('reset_password.html', token=token, csrf_token=csrf)
    if not ok_csrf(request.form.get('csrf_token','')):
        flash('Invalid request.','error'); return redirect(url_for('login_page'))
    pw  = request.form.get('password','')
    cpw = request.form.get('confirm_password','')
    if not vld_pw(pw):
        flash('Password: 8+ chars, 1 uppercase, 1 number.','error')
        return render_template('reset_password.html', token=token, csrf_token=csrf)
    if pw!=cpw:
        flash('Passwords do not match.','error')
        return render_template('reset_password.html', token=token, csrf_token=csrf)
    em = use_rst(token)
    if not em:
        flash('Reset link expired or already used.','error'); return redirect(url_for('login_page'))
    flash('Password updated! Please log in.','success') if update_password(em,pw) else flash('Update failed.','error')
    return redirect(url_for('login_page'))

@app.route('/logout')
def logout():
    session.clear(); flash('Signed out.','info'); return redirect(url_for('login_page'))

# ── Protected app routes ──────────────────────────────────────────────────────
@app.route('/')
@login_required
def index():
    return render_template('index.html', emotions=EMOTIONS,
                           user=session.get('user_fullname',''))

@app.route('/synthesize', methods=['POST'])
@login_required
def synthesize():
    if not rate_ok(request.remote_addr,20,60):
        return jsonify({'error':'Rate limit exceeded.'}),429
    try:
        d=request.get_json(force=True)
        text=d.get('text','').strip(); emo=d.get('emotion','neutral')
        if not text: return jsonify({'error':'No text provided'}),400
        return jsonify({'success':True,**synthesize_audio(text,emo)})
    except Exception as e:
        return jsonify({'error':str(e),'trace':traceback.format_exc()}),500

@app.route('/upload', methods=['POST'])
@login_required
def upload():
    if not rate_ok(request.remote_addr,10,60):
        return jsonify({'error':'Rate limit exceeded.'}),429
    try:
        if 'file' not in request.files: return jsonify({'error':'No file uploaded'}),400
        f=request.files['file']; emo=request.form.get('emotion','neutral')
        ext=Path(f.filename).suffix.lower()
        if ext not in {'.txt','.pdf','.docx'}: return jsonify({'error':f'Unsupported: {ext}'}),400
        tmp=os.path.join(OUTPUT_DIR,f'upload_{uuid.uuid4().hex}{ext}')
        f.save(tmp)
        text=(extract_pdf(tmp) if ext=='.pdf' else extract_docx(tmp) if ext=='.docx' else extract_txt(tmp))
        os.remove(tmp)   # file deleted immediately — never stored in cloud
        if not text.strip(): return jsonify({'error':'No text extracted'}),400
        return jsonify({'success':True,**synthesize_audio(text,emo)})
    except Exception as e:
        return jsonify({'error':str(e),'trace':traceback.format_exc()}),500

@app.route('/audio/<filename>')
@login_required
def serve_audio(filename):
    filename=os.path.basename(filename)
    if not re.match(r'^tts_[a-z]+_[a-f0-9]+\.wav$', filename): abort(403)
    fp=os.path.join(OUTPUT_DIR,filename)
    if not os.path.exists(fp): return jsonify({'error':'Not found'}),404
    with open(fp,'rb') as f: data=f.read()
    resp=Response(data,mimetype='audio/wav')
    resp.headers.update({'Content-Length':len(data),'Accept-Ranges':'bytes',
                         'Cache-Control':'no-cache, no-store, must-revalidate',
                         'Access-Control-Allow-Origin':'*'})
    return resp

@app.errorhandler(404)
def e404(e): return render_template('login.html',csrf_token=gen_csrf()),404
@app.errorhandler(403)
def e403(e): return jsonify({'error':'Forbidden'}),403

if __name__ == '__main__':
    print('🎙  VoxEmotion  v4  |  Auth + Security')
    print(f'   Firebase  : {"ON" if _USE_FIREBASE else "LOCAL (users.json)"}')
    print(f'   Dataset   : {DATASET_ROOT}')
    _load_t2()
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
