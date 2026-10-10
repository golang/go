@echo off
setlocal

where py.exe >nul 2>nul
if not errorlevel 1 goto use_py

where python.exe >nul 2>nul
if errorlevel 1 goto no_python
python "%~dpn0.py" %*
exit /b %ERRORLEVEL%

:use_py
py -3 "%~dpn0.py" %*
exit /b %ERRORLEVEL%

:no_python
echo This launcher requires Python 3. Install Python or use the matching .exe launcher. 1>&2
exit /b 9009
