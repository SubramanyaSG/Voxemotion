"""
config.py
=========
Central configuration for VoxEmotion.
Edit the paths in this file to match your system.
"""

import os
import secrets

# ── Base directory (project root) ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# ⚙️  EDIT THESE PATHS TO MATCH YOUR SYSTEM
# ============================================================
DATASET_ROOT = r'D:\7th Sem\Major Project - 2\Code\ets\Emotion Speech Dataset1\English'
OUTPUT_DIR   = os.path.join(BASE_DIR, 'outputs')
MODEL_DIR    = os.path.join(BASE_DIR, 'models')
# ============================================================

# ── Audio settings ────────────────────────────────────────────────────────────
SAMPLE_RATE  = 22050
HOP_LENGTH   = 256
N_MELS       = 80
N_FFT        = 1024
WIN_LENGTH   = 1024
MAX_FRAMES   = 300
EMOTIONS     = ['angry', 'happy', 'neutral', 'sad', 'surprise']

# ── Tacotron2 settings ────────────────────────────────────────────────────────
T2_MAX_CHARS    = 150      # max characters per synthesis chunk
SILENCE_MS      = 180      # ms of silence between sentence chunks
SILENCE_SAMPLES = int(SAMPLE_RATE * SILENCE_MS / 1000)

# ── Emotion prosody transform parameters ─────────────────────────────────────
# (pitch_semitones, energy_scale, speed_rate)
EMOTION_PARAMS = {
    'angry'   : ( 2.0, 1.30, 1.05),
    'happy'   : ( 3.5, 1.20, 1.10),
    'neutral' : ( 0.0, 1.00, 1.00),
    'sad'     : (-3.0, 0.75, 0.88),
    'surprise': ( 4.5, 1.15, 1.15),
}

# ── Flask settings ────────────────────────────────────────────────────────────
SECRET_KEY   = os.environ.get('FLASK_SECRET_KEY', secrets.token_hex(32))
PORT         = int(os.environ.get('PORT', 5000))
DEBUG        = False

# ── Firebase (optional – leave empty to use local users.json) ─────────────────
FIREBASE_CREDENTIALS = os.path.join(BASE_DIR, 'firebase_credentials.json')

# ── SMTP Email (optional – for password reset emails) ─────────────────────────
SMTP_EMAIL    = os.environ.get('SMTP_EMAIL',        'support_voxemotion@gmail.com')
SMTP_PASSWORD = os.environ.get('SMTP_APP_PASSWORD', '')  # Gmail App Password
APP_BASE_URL  = os.environ.get('APP_BASE_URL',      f'http://127.0.0.1:{PORT}')

# ── Ensure output directories exist ──────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)
