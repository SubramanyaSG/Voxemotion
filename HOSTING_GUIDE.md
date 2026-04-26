# 🌐 VoxEmotion – Free Hosting Guide

Deploy VoxEmotion to the internet for free using any of the options below.
All options provide a public HTTPS URL that anyone can access.

---

## Prerequisites Before Hosting

### 1. Push project to GitHub (required for all options)

```bash
# Install Git from https://git-scm.com
git init
git add .
git commit -m "VoxEmotion v4 initial commit"
git branch -M main

# Create a new repo on https://github.com (click + → New repository)
# Name it: voxemotion
# Do NOT initialise with README (your project already has one)

git remote add origin https://github.com/YOUR_USERNAME/voxemotion.git
git push -u origin main
```

### 2. Add a .gitignore file (IMPORTANT – protect secrets)

Create a file named `.gitignore` in your project root:

```
# Secrets – NEVER commit these
firebase_credentials.json
users.json
*.env
.env

# Python
venv/
__pycache__/
*.pyc
*.pyo
*.pyd
.Python

# Outputs (large audio files)
outputs/*.wav
outputs/*.png
models/*.pth

# Jupyter checkpoints
.ipynb_checkpoints/

# OS
.DS_Store
Thumbs.db
```

### 3. Create a Procfile (required for Render/Railway)

Create a file named `Procfile` (no extension) in project root:

```
web: python app.py
```

### 4. Set these Environment Variables on your hosting platform

| Variable | Value | Required |
|---|---|---|
| `FLASK_SECRET_KEY` | Any random 64-char string | **YES** |
| `SMTP_EMAIL` | `support_voxemotion@gmail.com` | Optional |
| `SMTP_APP_PASSWORD` | Your Gmail App Password | Optional (for email) |
| `APP_BASE_URL` | `https://your-app-url.com` | YES (for reset links) |
| `DATASET_ROOT` | Path to ESD dataset | YES |
| `OUTPUT_DIR` | Path for audio outputs | YES |
| `MODEL_DIR` | Path for model files | YES |

To generate a secure FLASK_SECRET_KEY, run this in Python:
```python
import secrets
print(secrets.token_hex(32))
```

---

## Option 1 — Render.com (RECOMMENDED – Best Free Tier)

**Free tier:** 750 hours/month, automatic HTTPS, custom domain support

```
Step 1: Go to https://render.com → Sign up with GitHub

Step 2: Click "New +" → "Web Service"

Step 3: Connect your GitHub repository "voxemotion"

Step 4: Configure:
   Name:           voxemotion
   Region:         Singapore (closest to India)
   Branch:         main
   Runtime:        Python 3
   Build Command:  pip install -r requirements.txt
   Start Command:  python app.py

Step 5: Add Environment Variables (click "Advanced"):
   FLASK_SECRET_KEY  = <your 64-char random key>
   APP_BASE_URL      = https://voxemotion.onrender.com
   DATASET_ROOT      = /opt/render/project/src/Emotion_Speech_Dataset
   OUTPUT_DIR        = /opt/render/project/src/outputs
   MODEL_DIR         = /opt/render/project/src/models
   SMTP_APP_PASSWORD = <your gmail app password>

Step 6: Click "Create Web Service"

Step 7: Wait 3-5 minutes for build to complete

Step 8: Your app is live at:  https://voxemotion.onrender.com
```

> **Note about dataset on Render:** The ESD dataset (11+ GB) is too large to push to GitHub.
> For demo purposes without the dataset, the app will use silent audio as fallback.
> For full functionality, use Railway with a persistent disk (see Option 2).

---

## Option 2 — Railway.app (Best for Persistent Storage)

**Free tier:** $5 credit/month (covers ~500 hours), persistent disk available

```
Step 1: Go to https://railway.app → Login with GitHub

Step 2: New Project → Deploy from GitHub Repo → select "voxemotion"

Step 3: Railway auto-detects Python and starts deploying

Step 4: Click your service → Variables → Add all environment variables

Step 5: For persistent storage (to keep the dataset):
   New → Volume → Mount path: /data
   Update environment variables:
   DATASET_ROOT = /data/English
   OUTPUT_DIR   = /data/outputs
   MODEL_DIR    = /data/models

Step 6: Settings → Domains → Generate Domain
   Your URL: https://voxemotion-production.up.railway.app

Step 7: Upload your dataset via Railway's shell:
   Click "Shell" tab → drag and drop dataset files
```

---

## Option 3 — PythonAnywhere (Easiest for Flask)

**Free tier:** Always-on web app, 512 MB storage, Python support built-in

```
Step 1: Go to https://www.pythonanywhere.com → Create free account

Step 2: Dashboard → Files → Upload Files
   Upload your project as a ZIP, then unzip it:
   Open Bash console:
   $ unzip tts_emotion_project_v2.zip
   $ cd tts_emotion_project_v2

Step 3: Install packages:
   $ pip3 install --user -r requirements.txt

Step 4: Web tab → Add a new web app
   → Manual configuration → Python 3.10

Step 5: Edit WSGI configuration file:
   Find the WSGI file path (shown on Web tab)
   Replace contents with:

   import sys
   sys.path.insert(0, '/home/YOUR_USERNAME/tts_emotion_project_v2')
   from app import app as application

Step 6: Static files (on Web tab):
   URL: /static/     Directory: /home/YOUR_USERNAME/tts_emotion_project_v2/static

Step 7: Environment variables (Web tab → "Environment variables"):
   FLASK_SECRET_KEY = your_random_key
   APP_BASE_URL     = https://YOUR_USERNAME.pythonanywhere.com

Step 8: Reload Web App → visit:
   https://YOUR_USERNAME.pythonanywhere.com
```

---

## Option 4 — Koyeb (Generous Free Tier)

**Free tier:** 2 services, 512 MB RAM, always-on

```
Step 1: https://app.koyeb.com → Sign up

Step 2: Create App → GitHub → select "voxemotion"

Step 3: Builder: Buildpack (auto-detects Python)

Step 4: Run Command: python app.py

Step 5: Port: 5000

Step 6: Add all environment variables

Step 7: Deploy → Get your public URL
```

---

## Gmail App Password Setup (for password reset emails)

```
Step 1: Sign in to your Google Account
Step 2: Go to https://myaccount.google.com/security
Step 3: Enable 2-Factor Authentication (required)
Step 4: Search "App passwords" in Google Account
Step 5: Select app: Mail | Select device: Other → type "VoxEmotion"
Step 6: Click Generate → copy the 16-character password
Step 7: Set SMTP_APP_PASSWORD=<16-char password> in your hosting platform
```

---

## Firebase Credentials on Hosted Server

**IMPORTANT:** Never commit `firebase_credentials.json` to GitHub.

Instead, on your hosting platform:
1. Open `firebase_credentials.json` and copy its entire contents
2. In your hosting platform's environment variables, add:
   ```
   FIREBASE_CREDENTIALS_JSON = {"type":"service_account","project_id":...}
   ```
3. Modify `app.py` to load from env var instead of file:

```python
# Replace the init_firebase() file-reading code with:
import json, tempfile

def init_firebase():
    global _fb_app, _fb_db, _fb_auth_mod, _USE_FIREBASE
    
    # Try environment variable first (for hosted deployments)
    creds_json = os.environ.get('FIREBASE_CREDENTIALS_JSON')
    if creds_json:
        try:
            import firebase_admin
            from firebase_admin import credentials, firestore, auth as fb_auth
            cred_dict = json.loads(creds_json)
            # Write to temp file (firebase_admin requires a file)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json',
                                             delete=False) as tmp:
                json.dump(cred_dict, tmp)
                tmp_path = tmp.name
            cred = credentials.Certificate(tmp_path)
            os.remove(tmp_path)
            if not firebase_admin._apps:
                _fb_app = firebase_admin.initialize_app(cred)
            else:
                _fb_app = firebase_admin.get_app()
            _fb_db       = firestore.client()
            _fb_auth_mod = fb_auth
            _USE_FIREBASE = True
            print('Firebase connected via environment variable.')
            return True
        except Exception as e:
            print(f'Firebase env init failed: {e}')
    
    # Fall back to file
    creds_path = os.path.join(BASE_DIR, 'firebase_credentials.json')
    # ... rest of existing code
```

---

## Domain Name (Optional, Free)

Get a free custom domain for your VoxEmotion app:

```
1. Go to https://www.freenom.com
   Register a free .tk, .ml, .ga, or .cf domain
   Example: voxemotion.ml

2. Or use https://js.org for a .js.org subdomain (GitHub Pages only)

3. On Render.com:
   Settings → Custom Domains → Add your domain
   Follow DNS configuration instructions

4. Update APP_BASE_URL environment variable to your custom domain
```

---

## Security Checklist Before Going Live

- [ ] `firebase_credentials.json` is in `.gitignore` and NOT on GitHub
- [ ] `users.json` is in `.gitignore`  
- [ ] `FLASK_SECRET_KEY` is set to a random 64-char string in env vars
- [ ] `SESSION_COOKIE_SECURE = True` in `app.py` (set this when using HTTPS)
- [ ] `APP_BASE_URL` points to your actual HTTPS URL
- [ ] Gmail App Password is set (not your main Gmail password)
- [ ] Firebase Firestore rules are set to authenticated access only:
  ```
  rules_version = '2';
  service cloud.firestore {
    match /databases/{database}/documents {
      match /users/{userId} {
        allow read, write: if false;  // Only admin SDK can access
      }
    }
  }
  ```

---

## After Deployment – Test Checklist

```
✅ Visit your public URL → see login page
✅ Register a new account → 30s cooldown works
✅ Login with credentials → redirected to main app
✅ Cookie banner appears → can accept/decline
✅ Type text + select emotion + Generate Speech → audio plays
✅ Upload a .txt/.pdf/.docx → audio generated
✅ Forgot password → email received (or check console log)
✅ Reset password link works → can log in with new password
✅ Logout → redirected to login page
✅ Try accessing / without login → redirected to login page
```

---

*VoxEmotion v4 — Text-to-Speech with Emotion Control using Deep Learning*
*Support: support_voxemotion@gmail.com*
