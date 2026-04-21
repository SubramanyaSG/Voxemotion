@echo off
REM ============================================================
REM  VoxEmotion v2 – Windows Setup & Launch
REM  Fixes: No module named 'pkg_resources'
REM  Solution: librosa removed; uses soundfile + scipy only
REM ============================================================

echo.
echo  =======================================================
echo    VoxEmotion v2  –  TTS with Emotion Control
echo    pkg_resources-free build
echo  =======================================================
echo.

python --version 2>nul
if %errorlevel% neq 0 (
    echo  [ERROR] Python not found in PATH.
    echo  Install Python 3.10.11 from https://python.org
    pause & exit /b 1
)

if not exist "venv\" (
    echo  [1/5] Creating virtual environment...
    python -m venv venv
)

echo  [2/5] Activating virtual environment...
call venv\Scripts\activate.bat

echo  [3/5] Upgrading pip + setuptools ...
python -m pip install --upgrade pip setuptools wheel --quiet

echo  [4/5] Installing PyTorch (CPU) ...
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu --quiet

echo  [5/5] Installing remaining dependencies ...
pip install -r requirements.txt --quiet

echo.
echo  Registering Jupyter kernel ...
python -m ipykernel install --user --name=voxemotion --display-name "VoxEmotion v2 (Python 3.10)"

echo.
echo  =======================================================
echo   Launching Jupyter Notebook ...
echo   Open: notebooks/TTS_Emotion_Control.ipynb
echo   Kernel: VoxEmotion v2 (Python 3.10)
echo   Run ALL cells top to bottom.
echo  =======================================================
echo.

jupyter notebook --notebook-dir=. --no-browser
pause
