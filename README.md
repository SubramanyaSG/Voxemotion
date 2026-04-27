# 🎙️ VoxEmotion – VS Code Setup Guide

> Text-to-Speech with Emotion Control using Deep Learning

---

## Project Structure

```
voxemotion/
├── app.py                    ← Flask web server (START HERE after training)
├── train.py                  ← Run ONCE to train the emotion classifier
├── config.py                 ← All settings — edit DATASET_ROOT here
├── requirements.txt          ← All Python dependencies
├── .env.example              ← Copy to .env and fill your values
├── .vscode/
│   ├── launch.json           ← VS Code run configurations
│   └── settings.json         ← VS Code Python settings
├── utils/
│   ├── audio.py              ← Audio I/O and DSP (soundfile + scipy)
│   ├── text_utils.py         ← Text normalization and chunking
│   ├── auth.py               ← Auth, CSRF, rate limiting, Firebase
│   └── dataset.py            ← ESD dataset scanner and feature extraction
├── models/
│   ├── emotion_model.py      ← CNN-LSTM emotion classifier
│   ├── synthesizer.py        ← Tacotron2 + retrieval TTS
│   └── emotion_best.pth      ← Saved checkpoint (created after training)
├── templates/
│   ├── login.html            ← Login + Register + Forgot Password
│   ├── reset_password.html   ← Secure password reset
│   └── index.html            ← Main TTS app
├── static/
│   ├── css/
│   │   ├── auth.css          ← Auth pages styling
│   │   └── style.css         ← Main app styling
│   └── js/
│       ├── auth.js           ← Login validation + cookie banner
│       └── app.js            ← Waveform + player + transcript highlight
├── outputs/                  ← Generated WAV files + training plots
├── models/                   ← Saved model checkpoints
├── firebase_credentials.json ← YOU create this (optional)
└── users.json                ← Auto-created in LOCAL auth mode
```

---

## Step-by-Step Setup in VS Code

### Step 1 – Install Python 3.10.11

Download: https://www.python.org/downloads/release/python-31011/

During installation:
- ✅ Check **Add Python to PATH**
- ✅ Check **Install for all users**

Verify in terminal:
```
python --version
# Should show: Python 3.10.11
```

---

### Step 2 – Open Project in VS Code

```
1. Open VS Code
2. File → Open Folder → select the voxemotion folder
3. VS Code will detect .vscode/settings.json automatically
```

---

### Step 3 – Open Terminal in VS Code

```
Terminal → New Terminal   (or press Ctrl + `)
```

---

### Step 4 – Create Virtual Environment

```cmd
python -m venv venv
```

Activate it:
```cmd
venv\Scripts\activate
```

You should see **(venv)** in your terminal prompt.

VS Code will also ask: *"We noticed a new virtual environment. Select it as your interpreter?"* → Click **Yes**.

---

### Step 5 – Install PyTorch (CPU)

```cmd
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
```

> For NVIDIA GPU (CUDA 11.8):
> ```cmd
> pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
> ```

---

### Step 6 – Install All Other Dependencies

```cmd
pip install -r requirements.txt
```

---

### Step 7 – Edit config.py (Set Your Paths)

Open `config.py` and update this line to match where your ESD dataset is:

```python
DATASET_ROOT = r'D:\7th Sem\Major Project - 2\Code\ets\Emotion Speech Dataset1\English'
```

`OUTPUT_DIR` and `MODEL_DIR` will be created automatically inside the project folder.

---

### Step 8 – Train the Model (Run Once)

```cmd
python train.py
```

This will:
- Scan your ESD dataset and verify all audio files
- Extract mel-spectrogram features
- Train the CNN-LSTM emotion classifier for 30 epochs
- Save mel-spectrogram visualization to `outputs/mel_spectrograms.png`
- Save training curves to `outputs/training_curves.png`
- Save confusion matrix to `outputs/confusion_matrix.png`
- Save the best model checkpoint to `models/emotion_best.pth`

Training takes approximately **5–15 minutes** depending on your CPU.

---

### Step 9 – Start the Web App

```cmd
python app.py
```

You will see:
```
=======================================================
  VoxEmotion – Web App
=======================================================
  URL      : http://127.0.0.1:5000
  Press Ctrl+C to stop
=======================================================
```

Open your browser: **http://127.0.0.1:5000**

---

### Step 10 – Register and Use

1. You will see the **Login page**
2. Click **Create an account** → Register with your details
3. Log in with your email and password
4. The **cookie banner** will appear — accept as preferred
5. Type text or upload a file → select emotion → click **Generate Speech**

---

## Running from VS Code Run Button

You can also run directly from VS Code:

1. Open `app.py`
2. Press **F5** or click the ▶ **Run** button
3. Select **"Run app.py (Flask Web App)"** from the dropdown

To run `train.py`:
1. Press **F5** on `train.py`
2. Or select **"Run train.py (Train Model)"** from the Run menu

---

## Auth Modes

### Local Mode (No setup needed)
- Works out of the box
- Users stored in `users.json` in the project folder
- Perfect for local development and demo

### Firebase Mode (Optional — for cloud storage)
```
1. Go to https://console.firebase.google.com
2. Create project "voxemotion"
3. Project Settings → Service Accounts → Generate new private key
4. Save downloaded file as firebase_credentials.json in project root
5. Enable Firestore Database (test mode)
6. Enable Authentication → Email/Password provider
7. Restart app.py
```

---

## Password Reset Email Setup (Optional)

Without this, reset links are printed to the VS Code terminal.

```
1. Enable 2-Factor Authentication on your Gmail
2. Google Account → Security → App Passwords
3. Generate password for "VoxEmotion"
4. Copy the 16-character password
5. Set in your system environment variables OR create a .env file:
   SMTP_APP_PASSWORD=your_16_char_password
```

---

## Common Errors & Fixes

| Error | Fix |
|---|---|
| `No module named 'pkg_resources'` | `pip install setuptools>=69.0.0 --force-reinstall` |
| `No module named 'soxr'` | `pip install soxr==0.3.7` |
| `No module named 'bcrypt'` | `pip install bcrypt==4.1.2` |
| `No module named 'firebase_admin'` | `pip install firebase-admin==6.5.0` |
| `No module named 'unidecode'` | `pip install Unidecode==1.3.8` |
| `No module named 'inflect'` | `pip install inflect==7.3.1` |
| Port 5000 already in use | Change `PORT=5001` in `config.py` |
| Dataset not found | Update `DATASET_ROOT` in `config.py` |
| `venv\Scripts\activate` fails | Run: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` then retry |

---

## Quick Command Reference

```cmd
# Activate environment
venv\Scripts\activate

# Install all packages
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Train model (run once)
python train.py

# Start web app
python app.py

# Deactivate environment
deactivate
```

---

*Support: support_voxemotion@gmail.com*
