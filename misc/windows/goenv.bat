@echo off
setlocal
for /f %%i in ("%0") do set cwd=%%~dpi
cd /d %cwd%

:: sanity checks
if exist "%cwd%"\bin\6g.exe (
set GOARCH=amd64
goto ok
)

if exist "%cwd%"\bin\8g.exe (
set GOARCH=386
goto ok
)

echo Unable to find the Go compiler
echo This batch file must run from the root Go folder
pause
exit

:ok
set GOROOT=%cwd%
set GOBIN=%GOROOT%\bin
set PATH=%GOBIN%;%PATH%

@CMD /F:ON
endlocal

