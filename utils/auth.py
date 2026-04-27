"""
utils/auth.py
=============
Authentication: Firebase Firestore or local JSON fallback.
Password hashing (bcrypt), CSRF tokens, rate limiting,
registration cooldown, and password reset tokens.
"""

import os
import re
import json
import time
import hmac
import hashlib
import secrets
import smtplib
import ssl
from datetime import datetime, timezone
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from flask import session

from config import (BASE_DIR, FIREBASE_CREDENTIALS, SMTP_EMAIL,
                    SMTP_PASSWORD, APP_BASE_URL)

# ── Firebase (optional) ───────────────────────────────────────────────────────
_fb_app      = None
_fb_db       = None
_fb_auth_mod = None
USE_FIREBASE = False


def init_firebase() -> bool:
    global _fb_app, _fb_db, _fb_auth_mod, USE_FIREBASE
    if not os.path.exists(FIREBASE_CREDENTIALS):
        print('⚠  firebase_credentials.json not found → LOCAL mode (users.json)')
        return False
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore, auth as fba
        if not firebase_admin._apps:
            cred    = credentials.Certificate(FIREBASE_CREDENTIALS)
            _fb_app = firebase_admin.initialize_app(cred)
        else:
            _fb_app = firebase_admin.get_app()
        _fb_db       = firestore.client()
        _fb_auth_mod = fba
        USE_FIREBASE = True
        print('✅  Firebase Firestore connected.')
        return True
    except Exception as e:
        print(f'⚠  Firebase init failed: {e} → LOCAL mode')
        return False


# ── Local user store (fallback) ───────────────────────────────────────────────
_LOCAL_USERS_PATH = os.path.join(BASE_DIR, 'users.json')


def _load_users() -> dict:
    if os.path.exists(_LOCAL_USERS_PATH):
        with open(_LOCAL_USERS_PATH, 'r') as f:
            return json.load(f)
    return {}


def _save_users(users: dict) -> None:
    with open(_LOCAL_USERS_PATH, 'w') as f:
        json.dump(users, f, indent=2)


# ── Password hashing ──────────────────────────────────────────────────────────
try:
    import bcrypt

    def hash_password(pw: str) -> str:
        return bcrypt.hashpw(pw.encode(), bcrypt.gensalt(12)).decode()

    def check_password(pw: str, hashed: str) -> bool:
        return bcrypt.checkpw(pw.encode(), hashed.encode())

except ImportError:
    def hash_password(pw: str) -> str:
        salt = os.urandom(32).hex()
        h    = hashlib.pbkdf2_hmac('sha256', pw.encode(), salt.encode(), 310000)
        return f"{salt}${h.hex()}"

    def check_password(pw: str, hashed: str) -> bool:
        try:
            salt, h = hashed.split('$', 1)
            candidate = hashlib.pbkdf2_hmac('sha256', pw.encode(), salt.encode(), 310000)
            return hmac.compare_digest(candidate.hex(), h)
        except Exception:
            return False


# ── CSRF tokens ───────────────────────────────────────────────────────────────
def generate_csrf() -> str:
    token = secrets.token_hex(32)
    session['csrf_token'] = token
    return token


def validate_csrf(token: str) -> bool:
    stored = session.get('csrf_token')
    if not stored or not token:
        return False
    return hmac.compare_digest(stored, token)


# ── Rate limiter (in-memory per IP) ──────────────────────────────────────────
_rate_store: dict = {}


def is_rate_allowed(ip: str, limit: int, window: int) -> bool:
    """Returns True if request is allowed, False if rate-limited."""
    now   = time.time()
    calls = [t for t in _rate_store.get(ip, []) if now - t < window]
    if len(calls) >= limit:
        return False
    calls.append(now)
    _rate_store[ip] = calls
    return True


# ── Registration cooldown (30 seconds per IP) ────────────────────────────────
_reg_cooldown: dict = {}


def get_reg_cooldown(ip: str) -> int:
    """Returns seconds remaining in cooldown, or 0 if allowed."""
    return max(0, int(30 - (time.time() - _reg_cooldown.get(ip, 0))))


def set_reg_cooldown(ip: str) -> None:
    _reg_cooldown[ip] = time.time()


# ── Password reset tokens (15-min expiry, single-use) ────────────────────────
_reset_tokens: dict = {}


def create_reset_token(email: str) -> str:
    token = secrets.token_urlsafe(48)
    _reset_tokens[token] = {
        'email': email,
        'exp'  : time.time() + 900,
        'used' : False
    }
    return token


def consume_reset_token(token: str) -> str | None:
    """Returns email if token is valid and unused, else None."""
    rec = _reset_tokens.get(token)
    if not rec or rec['used'] or time.time() > rec['exp']:
        return None
    rec['used'] = True
    return rec['email']


def is_reset_token_valid(token: str) -> bool:
    rec = _reset_tokens.get(token)
    return bool(rec and not rec['used'] and time.time() <= rec['exp'])


# ── Email sending ─────────────────────────────────────────────────────────────
def send_email(to: str, subject: str, html_body: str) -> bool:
    """Send email via Gmail SMTP. Prints to console if SMTP not configured."""
    if not SMTP_PASSWORD:
        print(f'\n[EMAIL – SMTP not configured, showing in console]')
        print(f'  To      : {to}')
        print(f'  Subject : {subject}')
        print(f'  Link    : (see HTML body)')
        return True
    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From']    = f'VoxEmotion <{SMTP_EMAIL}>'
        msg['To']      = to
        msg.attach(MIMEText(html_body, 'html'))
        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=ctx) as s:
            s.login(SMTP_EMAIL, SMTP_PASSWORD)
            s.sendmail(SMTP_EMAIL, to, msg.as_string())
        return True
    except Exception as e:
        print(f'Email send error: {e}')
        return False


def send_password_reset_email(email: str, token: str) -> None:
    link = f'{APP_BASE_URL}/reset-password/{token}'
    html = f"""
    <div style="font-family:sans-serif;background:#0d0d1a;color:#e8e8f8;
                padding:2rem;border-radius:12px;max-width:480px">
      <h2 style="color:#6e56ff">VoxEmotion – Password Reset</h2>
      <p>You requested a password reset. Click below to set a new password.</p>
      <p><strong>This link expires in 15 minutes and can only be used once.</strong></p>
      <a href="{link}" style="display:inline-block;margin:1rem 0;padding:.8rem 1.5rem;
         background:#6e56ff;color:#fff;border-radius:8px;
         text-decoration:none;font-weight:600;">Reset My Password</a>
      <p style="color:#7878a8;font-size:.8rem">
        If you did not request this, please ignore this email.</p>
      <p style="color:#7878a8;font-size:.8rem">
        Support: support_voxemotion@gmail.com</p>
    </div>"""
    send_email(email, 'VoxEmotion – Reset Your Password', html)


# ── User validation helpers ───────────────────────────────────────────────────
def valid_email(email: str) -> bool:
    return bool(re.match(r'^[^\s@]+@[^\s@]+\.[^\s@]{2,}$', email))


def valid_password(pw: str) -> bool:
    return (len(pw) >= 8
            and bool(re.search(r'[A-Z]', pw))
            and bool(re.search(r'[0-9]', pw)))


def valid_fullname(name: str) -> bool:
    return bool(re.match(r"^[A-Za-z]+(?:[ '-][A-Za-z]+)*$", name))


def valid_dob(dob: str) -> bool:
    try:
        if not re.match(r'^\d{2}/\d{2}/\d{4}$', dob):
            return False
        dd, mm, yyyy = map(int, dob.split('/'))
        d   = datetime(yyyy, mm, dd)
        age = (datetime.now() - d).days / 365.25
        return 5 <= age <= 120
    except Exception:
        return False


# ── User CRUD ─────────────────────────────────────────────────────────────────
def user_exists(email: str) -> bool:
    if USE_FIREBASE:
        try:
            _fb_auth_mod.get_user_by_email(email)
            return True
        except Exception:
            return False
    return email in _load_users()


def create_user(email: str, password: str,
                fullname: str, dob: str) -> bool:
    pw_hash = hash_password(password)
    if USE_FIREBASE:
        try:
            fb_user = _fb_auth_mod.create_user(email=email, display_name=fullname)
            _fb_db.collection('users').document(fb_user.uid).set({
                'fullname' : fullname,
                'dob'      : dob,
                'email'    : email,
                'pw_hash'  : pw_hash,
                'created'  : datetime.now(timezone.utc).isoformat(),
                'role'     : 'user'
            })
            return True
        except Exception as e:
            print(f'Firebase create_user error: {e}')
            return False
    else:
        users = _load_users()
        users[email] = {
            'fullname': fullname,
            'dob'     : dob,
            'pw_hash' : pw_hash,
            'role'    : 'user',
            'created' : datetime.now().isoformat()
        }
        _save_users(users)
        return True


def verify_user(email: str, password: str) -> dict | None:
    """Returns user dict if credentials are valid, else None."""
    if USE_FIREBASE:
        try:
            fb_user = _fb_auth_mod.get_user_by_email(email)
            doc     = _fb_db.collection('users').document(fb_user.uid).get()
            if doc.exists:
                data = doc.to_dict()
                if check_password(password, data.get('pw_hash', '')):
                    return {
                        'email'   : email,
                        'fullname': data.get('fullname', ''),
                        'uid'     : fb_user.uid,
                        'role'    : data.get('role', 'user')
                    }
        except Exception:
            return None
    else:
        users = _load_users()
        rec   = users.get(email)
        if rec and check_password(password, rec.get('pw_hash', '')):
            return {
                'email'   : email,
                'fullname': rec.get('fullname', ''),
                'role'    : rec.get('role', 'user')
            }
    return None


def update_password(email: str, new_password: str) -> bool:
    pw_hash = hash_password(new_password)
    if USE_FIREBASE:
        try:
            fb_user = _fb_auth_mod.get_user_by_email(email)
            _fb_auth_mod.update_user(fb_user.uid, password=new_password)
            _fb_db.collection('users').document(fb_user.uid).update(
                {'pw_hash': pw_hash}
            )
            return True
        except Exception as e:
            print(f'Firebase update_password error: {e}')
            return False
    else:
        users = _load_users()
        if email in users:
            users[email]['pw_hash'] = pw_hash
            _save_users(users)
            return True
        return False
