:: Copyright 2012 The Go Authors. All rights reserved.
:: Use of this source code is governed by a BSD-style
:: license that can be found in the LICENSE file.
@echo off

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
gcc -O2 -Wall -Werror -o ../bin/tool/dist.exe -Icmd/dist %DEFGOROOT% cmd/dist/buf.c cmd/dist/build.c cmd/dist/buildgc.c cmd/dist/buildruntime.c cmd/dist/goc2c.c cmd/dist/main.c cmd/dist/windows.c
if errorlevel 1 goto fail
:: Echo with no arguments prints whether echo is turned on, so echo dot.
echo .

echo # Building compilers and Go bootstrap tool.
..\bin\tool\dist bootstrap -v
if errorlevel 1 goto fail
echo .

echo # Building packages and commands.
..\bin\tool\go_bootstrap clean std
if errorlevel 1 goto fail
..\bin\tool\go_bootstrap install -a -v std
if errorlevel 1 goto fail
del ..\bin\tool\go_bootstrap.exe
echo .

if "x%1"=="x--no-banner" goto nobanner
..\bin\tool\dist banner
:nobanner

goto end

:fail
set GOBUILDFAIL=1

:end
