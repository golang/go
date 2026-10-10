@echo off
setlocal EnableExtensions

rem Build the native Go distribution first by running src\make.bat.
rem This script needs a MinGW-w64 GCC on PATH for cgo and c-shared builds.
for %%I in ("%~dp0..") do set "GOROOT_SHARED=%%~fI"
set "GOROOT=%GOROOT_SHARED%"
set "GO_SHARED=%GOROOT_SHARED%\bin\go.exe"
set "SHARED_SRC=%GOROOT_SHARED%\src"
set "SHARED_PY_LAUNCHER=%SHARED_SRC%\cmd\sharedlauncher\python_launcher.py"
set "SHARED_BAT_LAUNCHER=%SHARED_SRC%\cmd\sharedlauncher\windows_launcher.bat"
set "SHARED_STAGE=%GOROOT_SHARED%\shared-stage"

if exist "%GOROOT_SHARED%\native\bin\go-native.exe" set "GO_SHARED=%GOROOT_SHARED%\native\bin\go-native.exe"
if not exist "%GO_SHARED%" (
  echo Go is not built. Run src\make.bat first. 1>&2
  exit /b 1
)
where gcc >nul 2>nul
if errorlevel 1 (
  echo MinGW-w64 gcc was not found on PATH. 1>&2
  exit /b 1
)

if exist "%SHARED_STAGE%" rmdir /s /q "%SHARED_STAGE%"
mkdir "%SHARED_STAGE%\bin" || exit /b 1
mkdir "%SHARED_STAGE%\pkg\tool\windows_amd64" || exit /b 1

set "GOOS=windows"
set "GOARCH=amd64"
set "CGO_ENABLED=1"
set "GOFLAGS="
set "CC=gcc"
pushd "%SHARED_SRC%" || exit /b 1

for %%T in (go gofmt) do call :buildbin %%T || goto :failed
for %%T in (asm cgo compile link preprofile cover export fix vet) do call :buildtool %%T || goto :failed

popd
call :install go bin || exit /b 1
call :install gofmt bin || exit /b 1
for %%T in (asm cgo compile link preprofile cover export fix vet) do call :install %%T pkg\tool\windows_amd64 || exit /b 1

echo Shared Go toolchain installed under %GOROOT_SHARED%.
echo Native executables are preserved under %GOROOT_SHARED%\native.
exit /b 0

:buildbin
set "SHARED_TOOL=%~1"
"%GO_SHARED%" build -tags=sharedtool -buildmode=c-shared -o "%SHARED_STAGE%\bin\%SHARED_TOOL%.dll" "cmd/%SHARED_TOOL%"
if errorlevel 1 exit /b 1
del "%SHARED_STAGE%\bin\%SHARED_TOOL%.h" >nul 2>nul
exit /b 0

:buildtool
set "SHARED_TOOL=%~1"
"%GO_SHARED%" build -tags=sharedtool -buildmode=c-shared -o "%SHARED_STAGE%\pkg\tool\windows_amd64\%SHARED_TOOL%.dll" "cmd/%SHARED_TOOL%"
if errorlevel 1 exit /b 1
del "%SHARED_STAGE%\pkg\tool\windows_amd64\%SHARED_TOOL%.h" >nul 2>nul
exit /b 0

:install
set "SHARED_TOOL=%~1"
set "SHARED_DIR=%~2"
if not exist "%GOROOT_SHARED%\native\%SHARED_DIR%" mkdir "%GOROOT_SHARED%\native\%SHARED_DIR%"
if not exist "%GOROOT_SHARED%\native\%SHARED_DIR%\%SHARED_TOOL%-native.exe" (
  move /y "%GOROOT_SHARED%\%SHARED_DIR%\%SHARED_TOOL%.exe" "%GOROOT_SHARED%\native\%SHARED_DIR%\%SHARED_TOOL%-native.exe" >nul || exit /b 1
) else (
  del "%GOROOT_SHARED%\%SHARED_DIR%\%SHARED_TOOL%.exe" >nul 2>nul
)
copy /y "%SHARED_STAGE%\%SHARED_DIR%\%SHARED_TOOL%.dll" "%GOROOT_SHARED%\%SHARED_DIR%\%SHARED_TOOL%.dll" >nul || exit /b 1
copy /y "%SHARED_PY_LAUNCHER%" "%GOROOT_SHARED%\%SHARED_DIR%\%SHARED_TOOL%.py" >nul || exit /b 1
copy /y "%SHARED_BAT_LAUNCHER%" "%GOROOT_SHARED%\%SHARED_DIR%\%SHARED_TOOL%.bat" >nul || exit /b 1
exit /b 0

:failed
set "SHARED_STATUS=%ERRORLEVEL%"
popd
exit /b %SHARED_STATUS%
