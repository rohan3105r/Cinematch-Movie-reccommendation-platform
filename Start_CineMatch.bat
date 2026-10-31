@echo off
setlocal
cd /d "%~dp0"
title CineMatch Launcher

set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo Creating the project Python environment...
    where py >nul 2>nul
    if not errorlevel 1 (
        py -3 -m venv .venv
    ) else (
        where python >nul 2>nul
        if errorlevel 1 (
            echo ERROR: Python 3.10 or later is required but was not found.
            echo Install Python from https://www.python.org/downloads/ and try again.
            goto failed
        )
        python -m venv .venv
    )
    if errorlevel 1 (
        echo ERROR: Could not create the virtual environment.
        goto failed
    )
)

if not exist "%PYTHON_EXE%" (
    echo ERROR: The project Python environment is incomplete.
    goto failed
)

"%PYTHON_EXE%" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"
if errorlevel 1 (
    echo ERROR: CineMatch requires Python 3.10 or later.
    echo Install Python 3.10+ and recreate the .venv folder.
    goto failed
)

"%PYTHON_EXE%" -c "import numpy, pandas, sklearn, matplotlib, streamlit, docx, requests" >nul 2>nul
if errorlevel 1 (
    echo Installing or repairing project dependencies. Internet access may be required...
    "%PYTHON_EXE%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Dependency installation failed. Check your internet connection and try again.
        goto failed
    )
)

if not exist "data\demo\movies.csv" (
    echo Creating the bundled offline demo dataset...
    "%PYTHON_EXE%" scripts\generate_demo_data.py
    if errorlevel 1 (
        echo ERROR: Could not generate the demo dataset.
        goto failed
    )
)
if not exist "data\demo\ratings.csv" (
    echo Creating the bundled offline demo dataset...
    "%PYTHON_EXE%" scripts\generate_demo_data.py
    if errorlevel 1 (
        echo ERROR: Could not generate the demo dataset.
        goto failed
    )
)

"%PYTHON_EXE%" -m py_compile app.py launch_app.py
if errorlevel 1 (
    echo ERROR: Python found a syntax error in the application.
    goto failed
)

echo Starting CineMatch. Keep this window open while you use the app.
"%PYTHON_EXE%" launch_app.py
if errorlevel 1 (
    echo.
    echo ERROR: CineMatch stopped unexpectedly. Review the error shown above.
    goto failed
)
exit /b 0

:failed
echo.
pause
exit /b 1
