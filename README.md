# 🎙️ VoxEmotion v4 – Text-to-Speech with Emotion Control

> **Tacotron2 · CNN-LSTM · Firebase Auth · Flask · ESD Dataset**

---

## What's New in v4

| Feature | Details |
|---|---|
| Login / Register | Email + password auth with bcrypt hashing |
| Firebase Firestore | User profiles stored in free cloud (only you have admin access) |
| CSRF Protection | All forms protected against cross-site request forgery |
| Rate Limiting | 10 login/min, 3 registrations/hour, 3 resets/5 min per IP |
| Registration Cooldown | 30-second wait between registrations (prevents server crash) |
| Forgot Password | Time-limited (15 min), single-use secure reset tokens |
| Cookie Banner | Accept all / essential only — stored in localStorage |
| Security Headers | X-Frame-Options, CSP, XSS-Protection, nosniff |
| Support Footer | support_voxemotion@gmail.com on all pages |
| Local Fallback | Works without Firebase — users stored in users.json |
| Free Hosting Ready | Procfile + runtime.txt + .gitignore + HOSTING_GUIDE.md included |

---

## Python Version

**Use Python 3.10.11 (64-bit)**
Download: https://www.python.org/downloads/release/python-31011/

---

## Quick Start (Local)

```cmd
# 1. Double-click setup_and_run.bat
# 2. Open notebooks/TTS_Emotion_Control.ipynb
# 3. Run all 13 cells
# 4. Visit http://127.0.0.1:5000
# 5. Register → Login → Use VoxEmotion
```

---

## Install New Packages (v4 additions)

```cmd
conda activate voxemotion
pip install bcrypt==4.1.2
pip install firebase-admin==6.5.0
```

---

## Firebase Setup (Free)

```
1. https://console.firebase.google.com → Create project "voxemotion"
2. Project Settings → Service Accounts → Generate new private key
3. Save as firebase_credentials.json in this folder
4. Firestore Database → Create database → Start in test mode
5. Authentication → Sign-in method → Email/Password → Enable
```

Without `firebase_credentials.json`, the app runs in **LOCAL mode** — users are saved to `users.json` (works perfectly for local/demo use).

---

## Gmail Password Reset (Optional)

```
1. Enable 2FA on your Google account
2. Google Account → Security → App passwords
3. Generate password for "VoxEmotion"
4. Set environment variable: SMTP_APP_PASSWORD=<16-char password>
```

Without this, reset links are printed to the terminal console.

---

## Free Hosting

See **HOSTING_GUIDE.md** for complete step-by-step instructions for:
- Render.com (recommended)
- Railway.app (best for dataset storage)
- PythonAnywhere (simplest)
- Koyeb

---

## Project Structure

```
tts_emotion_project_v2/
├── app.py                     ← Flask backend (auth + TTS + security)
├── requirements.txt           ← All dependencies
├── setup_and_run.bat          ← Windows one-click setup
├── Procfile                   ← For Render/Railway deployment
├── runtime.txt                ← Python version for hosting platforms
├── .gitignore                 ← Protects secrets from GitHub
├── HOSTING_GUIDE.md           ← Complete free hosting instructions
├── runtime_config.json        ← Dataset/output paths
├── firebase_credentials.json  ← (YOU create this – not in repo)
├── users.json                 ← (Auto-created in LOCAL mode)
├── notebooks/
│   └── TTS_Emotion_Control.ipynb
├── templates/
│   ├── login.html             ← Login + Register + Forgot Password
│   ├── reset_password.html    ← Secure password reset page
│   └── index.html             ← Main TTS app (with support footer)
├── static/
│   ├── css/
│   │   ├── auth.css           ← Auth pages dark glassmorphism style
│   │   └── style.css          ← Main app style
│   └── js/
│       ├── auth.js            ← Validation, cookie banner, password strength
│       └── app.js             ← Waveform, player, transcript highlight
├── models/                    ← Saved CNN-LSTM checkpoint
└── outputs/                   ← Generated audio files
```

---

## Security Summary

- Passwords hashed with **bcrypt** (cost factor 12)
- **CSRF tokens** on every POST form
- **Rate limiting** per IP in memory
- **Session cookies** HttpOnly + SameSite=Lax
- **Path traversal** prevented on audio serving
- **Email enumeration** prevented in forgot-password flow
- **Reset tokens** 48-byte random, 15-min expiry, single-use
- **Security headers** on every HTTP response

---

*Support: support_voxemotion@gmail.com*
