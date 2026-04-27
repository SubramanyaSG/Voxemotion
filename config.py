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


def _load_dotenv_file() -> None:
    """Load key=value pairs from .env without requiring external packages."""
    env_path = os.path.join(BASE_DIR, '.env')
    if not os.path.exists(env_path):
        return

    try:
        with open(env_path, 'r', encoding='utf-8') as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception as e:
        print(f'Warning: could not load .env file: {e}')


_load_dotenv_file()

# ============================================================
# ⚙️  EDIT THESE PATHS TO MATCH YOUR SYSTEM
# ============================================================
# On hosted server, dataset is NOT present — that's fine
# Synthesis uses Tacotron2, not the dataset directly
DATASET_ROOT = os.environ.get('DATASET_ROOT', '')

# Use relative paths so they work on any server
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
MODEL_DIR  = os.path.join(BASE_DIR, 'models')
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
SMTP_PASSWORD = (os.environ.get('SMTP_APP_PASSWORD')
                 or os.environ.get('SMTP_PASSWORD', ''))  # Gmail App Password
APP_BASE_URL  = os.environ.get('APP_BASE_URL',      f'http://127.0.0.1:{PORT}')

# ── Ensure output directories exist ──────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)
