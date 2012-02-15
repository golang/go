:: Copyright 2012 The Go Authors. All rights reserved.
:: Use of this source code is governed by a BSD-style
:: license that can be found in the LICENSE file.
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

:: Grab default GOROOT_FINAL and set GOROOT for build.
:: The expression %VAR:\=\\% means to take %VAR%
:: and apply the substitution \ = \\, escaping the
:: backslashes.  Then we wrap that in quotes to create
:: a C string.
cd ..
set GOROOT=%CD%
cd src
if "x%GOROOT_FINAL%"=="x" set GOROOT_FINAL=%GOROOT%
set DEFGOROOT=-DGOROOT_FINAL="\"%GOROOT_FINAL:\=\\%\""

echo # Building C bootstrap tool.
echo cmd/dist
if not exist ..\bin\tool mkdir ..\bin\tool
:: Windows has no glob expansion, so spell out cmd/dist/*.c.
gcc -O2 -Wall -Werror -o cmd/dist/dist.exe -Icmd/dist %DEFGOROOT% cmd/dist/buf.c cmd/dist/build.c cmd/dist/buildgc.c cmd/dist/buildruntime.c cmd/dist/goc2c.c cmd/dist/main.c cmd/dist/windows.c
if errorlevel 1 goto fail
.\cmd\dist\dist env -wp >env.bat
if errorlevel 1 goto fail
call env.bat
del env.bat
:: Echo with no arguments prints whether echo is turned on, so echo dot.
echo .

echo # Building compilers and Go bootstrap tool.
.\cmd\dist\dist bootstrap -a -v
if errorlevel 1 goto fail
:: Delay move of dist tool to now, because bootstrap cleared tool directory.
move .\cmd\dist\dist.exe %GOTOOLDIR%\dist.exe
%GOTOOLDIR%\go_bootstrap clean -i std
echo .

if not %GOHOSTARCH% == %GOARCH% goto localbuild
if not %GOHOSTOS% == %GOOS% goto localbuild
goto mainbuild

:localbuild
echo # Building tools for local system. %GOHOSTOS%/%GOHOSTARCH%
setlocal
set GOOS=%GOHOSTOS%
set GOARCH=%GOHOSTARCH%
%GOTOOLDIR%\go_bootstrap install -v std
endlocal
if errorlevel 1 goto fail
echo .

:mainbuild
echo # Building packages and commands.
%GOTOOLDIR%\go_bootstrap install -a -v std
if errorlevel 1 goto fail
del %GOTOOLDIR%\go_bootstrap.exe
echo .

if x%1==x--no-banner goto nobanner
%GOTOOLDIR%\dist banner
:nobanner

goto end

:fail
set GOBUILDFAIL=1

:end
