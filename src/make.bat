:: Copyright 2012 The Go Authors. All rights reserved.
:: Use of this source code is governed by a BSD-style
:: license that can be found in the LICENSE file.

:: Environment variables that control make.bat:
::
:: GOROOT_FINAL: The expected final Go root, baked into binaries.
:: The default is the location of the Go tree during the build.
::
:: GOHOSTARCH: The architecture for host tools (compilers and
:: binaries).  Binaries of this type must be executable on the current
:: system, so the only common reason to set this is to set
:: GOHOSTARCH=386 on an amd64 machine.
::
:: GOARCH: The target architecture for installed packages and tools.
::
:: GOOS: The target operating system for installed packages and tools.
::
:: GO_GCFLAGS: Additional go tool compile arguments to use when
:: building the packages and commands.
::
:: GO_LDFLAGS: Additional go tool link arguments to use when
:: building the commands.
::
:: CGO_ENABLED: Controls cgo usage during the build. Set it to 1
:: to include all cgo related files, .c and .go file with "cgo"
:: build directive, in the build. Set it to 0 to ignore them.
::
:: CC: Command line to run to compile C code for GOHOSTARCH.
:: Default is "gcc".
::
:: CC_FOR_TARGET: Command line to run compile C code for GOARCH.
:: This is used by cgo. Default is CC.
::
:: FC: Command line to run to compile Fortran code.
:: This is used by cgo. Default is "gfortran".

@echo off

:: Keep environment variables within this script
:: unless invoked with --no-local.
if x%1==x--no-local goto nolocal
if x%2==x--no-local goto nolocal
setlocal
:nolocal

set GOBUILDFAIL=0

if exist make.bat goto ok
echo Must run make.bat from Go src directory.
goto fail 
:ok

:: Clean old generated file that will cause problems in the build.
del /F ".\pkg\runtime\runtime_defs.go" 2>NUL

:: Set GOROOT for build.
cd ..
set GOROOT=%CD%
cd src

echo ##### Building Go bootstrap tool.
echo cmd/dist
if not exist ..\bin\tool mkdir ..\bin\tool
if "x%GOROOT_BOOTSTRAP%"=="x" set GOROOT_BOOTSTRAP=%HOMEDRIVE%%HOMEPATH%\Go1.4
if not exist "%GOROOT_BOOTSTRAP%\bin\go.exe" goto bootstrapfail
setlocal
set GOROOT=%GOROOT_BOOTSTRAP%
set GOOS=
set GOARCH=
set GOBIN=
"%GOROOT_BOOTSTRAP%\bin\go" build -o cmd\dist\dist.exe .\cmd\dist
endlocal
if errorlevel 1 goto fail
.\cmd\dist\dist env -w -p >env.bat
if errorlevel 1 goto fail
call env.bat
del env.bat
echo.

if x%1==x--dist-tool goto copydist
if x%2==x--dist-tool goto copydist

set buildall=-a
if x%1==x--no-clean set buildall=
.\cmd\dist\dist bootstrap %buildall% -v
if errorlevel 1 goto fail
:: Delay move of dist tool to now, because bootstrap cleared tool directory.
move .\cmd\dist\dist.exe "%GOTOOLDIR%\dist.exe"
echo.

if not %GOHOSTARCH% == %GOARCH% goto localbuild
if not %GOHOSTOS% == %GOOS% goto localbuild
goto mainbuild

:localbuild
echo ##### Building packages and commands for host, %GOHOSTOS%/%GOHOSTARCH%.
:: CC_FOR_TARGET is recorded as the default compiler for the go tool. When building for the
:: host, however, use the host compiler, CC, from `cmd/dist/dist env` instead.
setlocal
set GOOS=%GOHOSTOS%
set GOARCH=%GOHOSTARCH%
"%GOTOOLDIR%\go_bootstrap" install -gcflags "%GO_GCFLAGS%" -ldflags "%GO_LDFLAGS%" -v std cmd
endlocal
if errorlevel 1 goto fail
echo.

:mainbuild
echo ##### Building packages and commands for %GOOS%/%GOARCH%.
setlocal
set CC=%CC_FOR_TARGET%
"%GOTOOLDIR%\go_bootstrap" install -gcflags "%GO_GCFLAGS%" -ldflags "%GO_LDFLAGS%" -a -v std cmd
endlocal
if errorlevel 1 goto fail
del "%GOTOOLDIR%\go_bootstrap.exe"
echo.

if x%1==x--no-banner goto nobanner
"%GOTOOLDIR%\dist" banner
:nobanner

goto end

:copydist
mkdir "%GOTOOLDIR%" 2>NUL
copy cmd\dist\dist.exe "%GOTOOLDIR%\"
goto end

:bootstrapfail
echo ERROR: Cannot find %GOROOT_BOOTSTRAP%\bin\go.exe
echo "Set GOROOT_BOOTSTRAP to a working Go tree >= Go 1.4."

:fail
set GOBUILDFAIL=1
if x%GOBUILDEXIT%==x1 exit %GOBUILDFAIL%

:end
