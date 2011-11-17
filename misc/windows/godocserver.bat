@echo off
setlocal
for /f "delims=" %%i in ('cd') do set cwd=%%i

if exist bin\godoc.exe goto ok
echo Unable to find the godoc executable
echo This batch file must run from the root Go folder
pause
exit

:ok
start bin\godoc -http=localhost:6060 -goroot="%cwd%"
start http://localhost:6060
endlocal
