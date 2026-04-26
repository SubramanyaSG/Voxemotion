@echo off
REM ============================================================
REM  VoxEmotion v4 – Windows Setup & Launch
REM  Includes: Auth, Firebase, bcrypt, Security
REM ============================================================

echo.
echo  =======================================================
echo    VoxEmotion v4  –  TTS with Emotion Control
echo    Auth + Firebase + Security Edition
echo  =======================================================
echo.

python --version 2>nul
if %errorlevel% neq 0 (
    echo  [ERROR] Python not found.
    echo  Install Python 3.10.11 from https://python.org
    pause & exit /b 1
)

if not exist "venv\" (
    echo  [1/6] Creating virtual environment...
    python -m venv venv
)

echo  [2/6] Activating virtual environment...
call venv\Scripts\activate.bat

echo  [3/6] Upgrading pip + setuptools...
python -m pip install --upgrade pip setuptools wheel --quiet

echo  [4/6] Installing PyTorch (CPU)...
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu --quiet

echo  [5/6] Installing all dependencies...
pip install -r requirements.txt --quiet

echo  [6/6] Registering Jupyter kernel...
python -m ipykernel install --user --name=voxemotion --display-name "VoxEmotion v4 (Python 3.10)"

echo.
echo  =======================================================
echo   IMPORTANT BEFORE FIRST RUN:
echo   1. Copy firebase_credentials.json to this folder
echo      (download from Firebase Console Service Accounts)
echo   2. Set environment variables (optional for email):
echo      SMTP_APP_PASSWORD=your_gmail_app_password
echo      APP_BASE_URL=http://127.0.0.1:5000
echo   3. Without Firebase the app works in LOCAL mode
echo      (users stored in users.json)
echo  =======================================================
echo.

echo  Launching Jupyter Notebook...
jupyter notebook --notebook-dir=. --no-browser

pause
