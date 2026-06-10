@echo off
echo.
echo ================================================
echo   Forensic AI Investigation System
echo ================================================
echo.

cd /d "%~dp0"

REM ══════════════════════════════════════════════════
REM  Step 1: Find Python
REM ══════════════════════════════════════════════════

set "PY_CMD="

where python >nul 2>&1
if not errorlevel 1 (
    python -c "import sys; exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>&1
    if not errorlevel 1 (
        set "PY_CMD=python"
        goto :found_python
    )
)

where py >nul 2>&1
if not errorlevel 1 (
    py -c "import sys; exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>&1
    if not errorlevel 1 (
        set "PY_CMD=py"
        goto :found_python
    )
)

where python3 >nul 2>&1
if not errorlevel 1 (
    python3 -c "import sys; exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>&1
    if not errorlevel 1 (
        set "PY_CMD=python3"
        goto :found_python
    )
)

echo.
echo [ERROR] Python 3.10+ not found. Install from https://www.python.org/downloads/
pause
exit /b 1

:found_python
for /f "tokens=*" %%A in ('%PY_CMD% -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"') do set "PY_VER=%%A"
echo [OK] Python %PY_VER% (%PY_CMD%)

REM ══════════════════════════════════════════════════
REM  Step 2: Setup .env file
REM ══════════════════════════════════════════════════

if not exist .env (
    if exist .env.example (
        echo [INFO] Creating .env from .env.example
        copy ".env.example" ".env" >nul
    ) else (
        echo [INFO] Creating .env
        echo SECRET_KEY=change-me-to-a-random-string> .env
        echo DEEPGRAM_API_KEY=>> .env
        echo GENERATION_MODE=api>> .env
        echo HF_API_TOKEN=>> .env
    )
    echo.
    echo ============================================================
    echo   .env created. Edit it and add your API keys, then re-run.
    echo ============================================================
    echo.
    pause
    exit /b 0
)

REM Read GENERATION_MODE from .env using Python to avoid batch parsing issues
for /f "tokens=*" %%A in ('%PY_CMD% -c "open('.env').read(); [print(l.split('=',1)[1].strip()) for l in open('.env') if l.strip().startswith('GENERATION_MODE')]"') do set "GEN_MODE=%%A"
if not defined GEN_MODE set "GEN_MODE=api"

echo [OK] Generation mode: %GEN_MODE%

REM ══════════════════════════════════════════════════
REM  Step 3: Virtual environment
REM ══════════════════════════════════════════════════

if exist venv (
    "venv\Scripts\python.exe" -c "print('ok')" >nul 2>&1
    if errorlevel 1 (
        echo [INFO] Existing venv is broken, recreating...
        rmdir /s /q venv
    )
)

if not exist venv (
    echo Creating virtual environment...
    %PY_CMD% -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
)

if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] venv is corrupted. Delete the venv folder and re-run.
    pause
    exit /b 1
)
call "venv\Scripts\activate.bat"

REM ══════════════════════════════════════════════════
REM  Step 4: Install dependencies
REM ══════════════════════════════════════════════════

echo.
python -m pip install --upgrade pip >nul 2>&1

echo Installing core packages...
pip install -r requirements.txt --progress-bar on
if errorlevel 1 (
    echo [ERROR] Failed to install core packages.
    pause
    exit /b 1
)
echo [OK] Core packages installed.

if /i "%GEN_MODE%"=="local" goto :install_local
goto :skip_local

:install_local
echo.
echo Installing local generation packages - torch is about 2.5 GB...
echo.
pip install -r requirements-local.txt --progress-bar on
if errorlevel 1 (
    echo [ERROR] Failed to install local generation packages.
    echo         Try GENERATION_MODE=api in .env instead.
    pause
    exit /b 1
)
echo [OK] Local generation packages installed.
goto :done_install

:skip_local
echo [OK] API mode - torch/diffusers not needed, skipping.

:done_install

REM ══════════════════════════════════════════════════
REM  Step 5: Quick health check
REM ══════════════════════════════════════════════════

echo.
python -c "import numpy, cv2, PIL, onnxruntime, flask; print('[OK] All core packages verified')"
if errorlevel 1 (
    echo [ERROR] Some core packages failed to import.
    pause
    exit /b 1
)

REM ══════════════════════════════════════════════════
REM  Step 6: Launch
REM ══════════════════════════════════════════════════

echo.
echo ================================================
echo   Starting Forensic AI System
echo   Browser: http://127.0.0.1:5000
echo   Login:   superadmin / Admin@123
echo   Mode:    %GEN_MODE%
echo   Stop:    Ctrl+C
echo ================================================
echo.

python app.py
set "APP_EXIT=%errorlevel%"

echo.
if not "%APP_EXIT%"=="0" (
    echo ================================================
    echo   App crashed. Check the error messages above.
    echo ================================================
)
echo.
echo Press any key to close...
pause >nul
