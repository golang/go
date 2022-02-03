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
if x%3==x--no-local goto nolocal
if x%4==x--no-local goto nolocal
setlocal
:nolocal

set GOENV=off
set GOBUILDFAIL=0
set GOFLAGS=
set GO111MODULE=

if exist make.bat goto ok
echo Must run make.bat from Go src directory.
goto fail
:ok

:: Clean old generated file that will cause problems in the build.
del /F ".\pkg\runtime\runtime_defs.go" 2>NUL

:: Set GOROOT for build.
cd ..
set GOROOT_TEMP=%CD%
set GOROOT=
cd src
set vflag=
if x%1==x-v set vflag=-v
if x%2==x-v set vflag=-v
if x%3==x-v set vflag=-v
if x%4==x-v set vflag=-v

if not exist ..\bin\tool mkdir ..\bin\tool

:: Calculating GOROOT_BOOTSTRAP
if not "x%GOROOT_BOOTSTRAP%"=="x" goto bootstrapset
for /f "tokens=*" %%g in ('where go 2^>nul') do (
	if "x%GOROOT_BOOTSTRAP%"=="x" (
		for /f "tokens=*" %%i in ('%%g env GOROOT 2^>nul') do (
			if /I not "%%i"=="%GOROOT_TEMP%" (
				set GOROOT_BOOTSTRAP=%%i
			)
		)
	)
)
if "x%GOROOT_BOOTSTRAP%"=="x" if exist "%HOMEDRIVE%%HOMEPATH%\go1.17" set GOROOT_BOOTSTRAP=%HOMEDRIVE%%HOMEPATH%\go1.17
if "x%GOROOT_BOOTSTRAP%"=="x" if exist "%HOMEDRIVE%%HOMEPATH%\sdk\go1.17" set GOROOT_BOOTSTRAP=%HOMEDRIVE%%HOMEPATH%\sdk\go1.17
if "x%GOROOT_BOOTSTRAP%"=="x" set GOROOT_BOOTSTRAP=%HOMEDRIVE%%HOMEPATH%\Go1.4

:bootstrapset
if not exist "%GOROOT_BOOTSTRAP%\bin\go.exe" goto bootstrapfail
set GOROOT=%GOROOT_TEMP%
set GOROOT_TEMP=

echo Building Go cmd/dist using %GOROOT_BOOTSTRAP%
if x%vflag==x-v echo cmd/dist
setlocal
set GOROOT=%GOROOT_BOOTSTRAP%
set GOOS=
set GOARCH=
set GOBIN=
set GO111MODULE=off
"%GOROOT_BOOTSTRAP%\bin\go.exe" build -o cmd\dist\dist.exe .\cmd\dist
endlocal
if errorlevel 1 goto fail
.\cmd\dist\dist.exe env -w -p >env.bat
if errorlevel 1 goto fail
call env.bat
del env.bat
if x%vflag==x-v echo.

if x%1==x--dist-tool goto copydist
if x%2==x--dist-tool goto copydist
if x%3==x--dist-tool goto copydist
if x%4==x--dist-tool goto copydist

set bootstrapflags=
if x%1==x--no-clean set bootstrapflags=--no-clean
if x%2==x--no-clean set bootstrapflags=--no-clean
if x%3==x--no-clean set bootstrapflags=--no-clean
if x%4==x--no-clean set bootstrapflags=--no-clean
if x%1==x--no-banner set bootstrapflags=%bootstrapflags% --no-banner
if x%2==x--no-banner set bootstrapflags=%bootstrapflags% --no-banner
if x%3==x--no-banner set bootstrapflags=%bootstrapflags% --no-banner
if x%4==x--no-banner set bootstrapflags=%bootstrapflags% --no-banner

:: Run dist bootstrap to complete make.bash.
:: Bootstrap installs a proper cmd/dist, built with the new toolchain.
:: Throw ours, built with Go 1.4, away after bootstrap.
.\cmd\dist\dist.exe bootstrap -a %vflag% %bootstrapflags%
if errorlevel 1 goto fail
del .\cmd\dist\dist.exe
goto end

:: DO NOT ADD ANY NEW CODE HERE.
:: The bootstrap+del above are the final step of make.bat.
:: If something must be added, add it to cmd/dist's cmdbootstrap,
:: to avoid needing three copies in three different shell languages
:: (make.bash, make.bat, make.rc).

:copydist
mkdir "%GOTOOLDIR%" 2>NUL
copy cmd\dist\dist.exe "%GOTOOLDIR%\"
goto end

:bootstrapfail
echo ERROR: Cannot find %GOROOT_BOOTSTRAP%\bin\go.exe
echo Set GOROOT_BOOTSTRAP to a working Go tree ^>= Go 1.4.

:fail
set GOBUILDFAIL=1
if x%GOBUILDEXIT%==x1 exit %GOBUILDFAIL%

:end
