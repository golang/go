:: Copyright 2012 The Go Authors. All rights reserved.
:: Use of this source code is governed by a BSD-style
:: license that can be found in the LICENSE file.

@echo off

if exist ..\bin\go.exe goto ok
echo Must run run.bat from Go src directory after installing cmd/go.
goto fail
:ok

:: Keep environment variables within this script
:: unless invoked with --no-local.
if x%1==x--no-local goto nolocal
if x%2==x--no-local goto nolocal
setlocal
:nolocal

set GOBUILDFAIL=0

set GOPATH=c:\nonexist-gopath
:: Issue 14340: ignore GOBIN during all.bat.
set GOBIN=
set GOFLAGS=
set GO111MODULE=

rem TODO avoid rebuild if possible

if x%1==x--no-rebuild goto norebuild
echo ##### Building packages and commands.
..\bin\go install -a -v std cmd
if errorlevel 1 goto fail
echo.
:norebuild

:: get CGO_ENABLED
..\bin\go env > env.bat
if errorlevel 1 goto fail
call env.bat
del env.bat
echo.

..\bin\go tool dist test
if errorlevel 1 goto fail
echo.

goto end

:fail
set GOBUILDFAIL=1

:end
