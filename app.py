"""
app.py
======
VoxEmotion Flask Web Application (VS Code / standalone Python version)

Run with:
    python app.py

Then open: http://127.0.0.1:5000
"""

import os
import re
import uuid
import traceback
from pathlib import Path
from datetime import timedelta
from functools import wraps

from flask import (Flask, request, jsonify, render_template,
                   Response, redirect, url_for, session, flash, abort)
from flask_cors import CORS

# ── Project imports ───────────────────────────────────────────────────────────
from config import (BASE_DIR, OUTPUT_DIR, MODEL_DIR, EMOTIONS,
                    SECRET_KEY, PORT, DEBUG)

from utils.auth import (
    init_firebase, generate_csrf, validate_csrf,
    is_rate_allowed, get_reg_cooldown, set_reg_cooldown,
    create_reset_token, consume_reset_token, is_reset_token_valid,
    send_password_reset_email,
    valid_email, valid_password, valid_fullname, valid_dob,
    user_exists, create_user, verify_user, update_password
)
from utils.dataset import get_dataset_df
from utils.text_utils import normalize_text, extract_text_from_file
from models.synthesizer import EmotionSynthesizer

import soundfile as sf

# ── Initialize Firebase ───────────────────────────────────────────────────────
init_firebase()

# ── Initialize Synthesizer (loads Tacotron2 once at startup) ─────────────────
print('Initializing synthesizer …')
_df         = get_dataset_df()
_synthesizer = EmotionSynthesizer(dataset_df=_df, output_dir=OUTPUT_DIR)
print('Synthesizer ready.\n')

# ── Flask application ─────────────────────────────────────────────────────────
app = Flask(__name__,
            template_folder=os.path.join(BASE_DIR, 'templates'),
            static_folder=os.path.join(BASE_DIR, 'static'))

app.secret_key = SECRET_KEY
app.config.update(
    SESSION_COOKIE_HTTPONLY   = True,
    SESSION_COOKIE_SAMESITE   = 'Lax',
    SESSION_COOKIE_SECURE     = False,   # Set True in production with HTTPS
    PERMANENT_SESSION_LIFETIME= timedelta(days=7),
    MAX_CONTENT_LENGTH        = 50 * 1024 * 1024,   # 50 MB
)
CORS(app, supports_credentials=True)


# ── Security headers on every response ───────────────────────────────────────
@app.after_request
def set_security_headers(response):
    response.headers['X-Content-Type-Options']  = 'nosniff'
    response.headers['X-Frame-Options']         = 'DENY'
    response.headers['X-XSS-Protection']        = '1; mode=block'
    response.headers['Referrer-Policy']         = 'strict-origin-when-cross-origin'
    response.headers['Permissions-Policy']      = 'microphone=(), camera=()'
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com "
        "https://fonts.gstatic.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "connect-src 'self'; "
        "img-src 'self' data:; "
        "media-src 'self' blob:;"
    )
    return response


# ── Login required decorator ──────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_email' not in session:
            flash('Please log in to continue.', 'info')
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated


# ════════════════════════════════════════════════════════════════════════════
# AUTH ROUTES
# ════════════════════════════════════════════════════════════════════════════

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    """Login page — also contains register and forgot-password panels."""
    if 'user_email' in session:
        return redirect(url_for('index'))

    if request.method == 'POST':
        # Validate against token minted on the previous page render.
        if not validate_csrf(request.form.get('csrf_token', '')):
            flash('Invalid request. Please try again.', 'error')
            return redirect(url_for('login_page'))

        ip = request.remote_addr
        if not is_rate_allowed(ip, limit=10, window=60):
            flash('Too many login attempts. Please wait 1 minute.', 'error')
            return render_template('login.html', csrf_token=generate_csrf())

        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')

        if not valid_email(email) or not password:
            flash('Please enter a valid email and password.', 'error')
            return render_template('login.html', csrf_token=generate_csrf())

        user = verify_user(email, password)
        if not user:
            flash('Incorrect email or password.', 'error')
            return render_template('login.html', csrf_token=generate_csrf())

        session.permanent = bool(request.form.get('remember'))
        session['user_email']    = user['email']
        session['user_fullname'] = user.get('fullname', '')
        session['user_role']     = user.get('role', 'user')
        session.pop('csrf_token', None)
        return redirect(url_for('index'))

    return render_template('login.html', csrf_token=generate_csrf())


@app.route('/register', methods=['POST'])
def register():
    """Handle new user registration."""
    if not validate_csrf(request.form.get('csrf_token', '')):
        flash('Invalid request. Please try again.', 'error')
        return redirect(url_for('login_page'))

    ip   = request.remote_addr
    wait = get_reg_cooldown(ip)
    if wait:
        flash(f'Please wait {wait} more seconds before registering again.', 'warning')
        return render_template('login.html', csrf_token=generate_csrf())

    if not is_rate_allowed(ip, limit=3, window=3600):
        flash('Registration limit reached. Please try again later.', 'error')
        return render_template('login.html', csrf_token=generate_csrf())

    fullname = request.form.get('fullname', '').strip()
    dob      = request.form.get('dob', '').strip()
    email    = request.form.get('email', '').strip().lower()
    password = request.form.get('password', '')
    confirm  = request.form.get('confirm_password', '')

    errors = []
    if not valid_fullname(fullname):
        errors.append("Name can include letters, spaces, hyphens, and apostrophes only.")
    if not valid_dob(dob):
        errors.append('Invalid date of birth. Use DD/MM/YYYY.')
    if not valid_email(email):
        errors.append('Invalid email address.')
    if not valid_password(password):
        errors.append('Password must be 8+ characters with 1 uppercase letter and 1 number.')
    if password != confirm:
        errors.append('Passwords do not match.')
    if not request.form.get('agree_terms'):
        errors.append('You must accept the Terms & Conditions.')

    if errors:
        for err in errors:
            flash(err, 'error')
        return render_template('login.html', csrf_token=generate_csrf())

    if user_exists(email):
        flash('An account with this email already exists.', 'error')
        return render_template('login.html', csrf_token=generate_csrf())

    if not create_user(email, password, fullname, dob):
        flash('Registration failed. Please try again.', 'error')
        return render_template('login.html', csrf_token=generate_csrf())

    set_reg_cooldown(ip)
    flash('Account created successfully! Please log in.', 'success')
    return redirect(url_for('login_page'))


@app.route('/forgot-password', methods=['POST'])
def forgot_password():
    """Send password reset email."""
    if not validate_csrf(request.form.get('csrf_token', '')):
        flash('Invalid request.', 'error')
        return redirect(url_for('login_page'))

    if not is_rate_allowed(request.remote_addr, limit=3, window=300):
        flash('Too many reset requests. Please wait 5 minutes.', 'error')
        return render_template('login.html', csrf_token=generate_csrf())

    email = request.form.get('email', '').strip().lower()
    if not valid_email(email):
        flash('Please enter a valid email address.', 'error')
        return render_template('login.html', csrf_token=generate_csrf())

    # Always show same message to prevent email enumeration
    if user_exists(email):
        token = create_reset_token(email)
        send_password_reset_email(email, token)

    flash('If that email is registered, a reset link has been sent.', 'success')
    return redirect(url_for('login_page'))


@app.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    """Password reset page (accessed via emailed link)."""
    if request.method == 'GET':
        if not is_reset_token_valid(token):
            flash('This reset link is invalid or has expired. Please request a new one.', 'error')
            return redirect(url_for('login_page'))
        return render_template('reset_password.html', token=token, csrf_token=generate_csrf())

    if not validate_csrf(request.form.get('csrf_token', '')):
        flash('Invalid request.', 'error')
        return redirect(url_for('login_page'))

    password = request.form.get('password', '')
    confirm  = request.form.get('confirm_password', '')

    if not valid_password(password):
        flash('Password must be 8+ characters with 1 uppercase letter and 1 number.', 'error')
        return render_template('reset_password.html', token=token, csrf_token=generate_csrf())

    if password != confirm:
        flash('Passwords do not match.', 'error')
        return render_template('reset_password.html', token=token, csrf_token=generate_csrf())

    email = consume_reset_token(token)
    if not email:
        flash('Reset link expired or already used. Please request a new one.', 'error')
        return redirect(url_for('login_page'))

    if update_password(email, password):
        flash('Password updated successfully! Please log in.', 'success')
    else:
        flash('Could not update password. Please try again.', 'error')

    return redirect(url_for('login_page'))


@app.route('/logout')
def logout():
    session.clear()
    flash('You have been signed out.', 'info')
    return redirect(url_for('login_page'))


# ════════════════════════════════════════════════════════════════════════════
# MAIN APP ROUTES  (protected — must be logged in)
# ════════════════════════════════════════════════════════════════════════════

@app.route('/')
@login_required
def index():
    return render_template(
        'index.html',
        emotions=EMOTIONS,
        user=session.get('user_fullname', '')
    )


@app.route('/synthesize', methods=['POST'])
@login_required
def synthesize():
    """Synthesize speech from typed text input."""
    if not is_rate_allowed(request.remote_addr, limit=20, window=60):
        return jsonify({'error': 'Rate limit exceeded. Please slow down.'}), 429

    try:
        data    = request.get_json(force=True)
        text    = data.get('text', '').strip()
        emotion = data.get('emotion', 'neutral')

        if not text:
            return jsonify({'error': 'No text provided.'}), 400

        result = _synthesizer.synthesize(text, emotion)
        return jsonify({'success': True, **result})

    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/upload', methods=['POST'])
@login_required
def upload():
    """Synthesize speech from uploaded file (.txt / .pdf / .docx)."""
    if not is_rate_allowed(request.remote_addr, limit=10, window=60):
        return jsonify({'error': 'Rate limit exceeded. Please slow down.'}), 429

    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded.'}), 400

        f       = request.files['file']
        emotion = request.form.get('emotion', 'neutral')
        ext     = Path(f.filename).suffix.lower()

        if ext not in {'.txt', '.pdf', '.docx'}:
            return jsonify({'error': f'Unsupported file type: {ext}'}), 400

        # Save temp file, extract text, then DELETE immediately
        # Files are never stored in cloud — only processed locally
        tmp = os.path.join(OUTPUT_DIR, f'upload_{uuid.uuid4().hex}{ext}')
        f.save(tmp)
        text = extract_text_from_file(tmp)
        os.remove(tmp)   # deleted right away

        if not text.strip():
            return jsonify({'error': 'No text could be extracted from this file.'}), 400

        result = _synthesizer.synthesize(text, emotion)
        return jsonify({'success': True, **result})

    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/audio/<filename>')
@login_required
def serve_audio(filename):
    """Serve generated audio file with proper headers."""
    # Sanitize — prevent path traversal attacks
    filename = os.path.basename(filename)
    if not re.match(r'^tts_[a-z]+_[a-f0-9]+\.wav$', filename):
        abort(403)

    filepath = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'Audio file not found.'}), 404

    with open(filepath, 'rb') as f:
        data = f.read()

    resp = Response(data, mimetype='audio/wav')
    resp.headers['Content-Length']              = len(data)
    resp.headers['Accept-Ranges']               = 'bytes'
    resp.headers['Cache-Control']               = 'no-cache, no-store, must-revalidate'
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp


# ── Error handlers ─────────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return render_template('login.html', csrf_token=generate_csrf()), 404

@app.errorhandler(403)
def forbidden(e):
    return jsonify({'error': 'Forbidden'}), 403

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 50 MB.'}), 413

@app.errorhandler(429)
def rate_limited(e):
    return jsonify({'error': 'Too many requests. Please slow down.'}), 429


# ════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('=' * 55)
    print('  🎙  VoxEmotion – Web App')
    print('=' * 55)
    print(f'  URL      : http://127.0.0.1:{PORT}')
    print(f'  Dataset  : {BASE_DIR}')
    print(f'  Outputs  : {OUTPUT_DIR}')
    print(f'  Press Ctrl+C to stop')
    print('=' * 55 + '\n')

    app.run(
        host='0.0.0.0',
        port=PORT,
        debug=DEBUG,
        use_reloader=False   # Must be False — prevents model double-loading
    )
