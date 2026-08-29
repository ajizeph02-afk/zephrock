@echo off
title Zephrock
setlocal enabledelayedexpansion

where python >nul 2>nul
if %errorlevel%==0 (
    set PYCMD=python
) else (
    where py >nul 2>nul
    if %errorlevel%==0 (
        set PYCMD=py
    ) else (
        echo Could not find Python on this machine. Install Python 3.9+ and try again.
        pause
        exit /b 1
    )
)

%PYCMD% -c "import streamlit" >nul 2>nul
if not %errorlevel%==0 (
    echo Installing required packages, first run only...
    %PYCMD% -m pip install -r requirements.txt
    if not %errorlevel%==0 (
        echo Package install failed. Check the messages above.
        pause
        exit /b 1
    )
)

%PYCMD% -m streamlit run app.py
if not %errorlevel%==0 (
    echo.
    echo Zephrock failed to start. See the message above.
    pause
)
