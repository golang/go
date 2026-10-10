@echo off
setlocal
for %%I in ("%~dp0..") do set "GOROOT_NATIVE=%%~fI"
for %%T in (go gofmt) do call :restore %%T bin || exit /b 1
for %%T in (asm cgo compile link preprofile cover export fix vet) do call :restore %%T pkg\tool\windows_amd64 || exit /b 1
echo Restored native Go executables from %GOROOT_NATIVE%\native.
exit /b 0

:restore
set "NATIVE_TOOL=%~1"
set "NATIVE_DIR=%~2"
if not exist "%GOROOT_NATIVE%\native\%NATIVE_DIR%\%NATIVE_TOOL%-native.exe" (
  echo Missing native executable: %GOROOT_NATIVE%\native\%NATIVE_DIR%\%NATIVE_TOOL%-native.exe 1>&2
  exit /b 1
)
copy /y "%GOROOT_NATIVE%\native\%NATIVE_DIR%\%NATIVE_TOOL%-native.exe" "%GOROOT_NATIVE%\%NATIVE_DIR%\%NATIVE_TOOL%.exe" >nul || exit /b 1
exit /b 0
