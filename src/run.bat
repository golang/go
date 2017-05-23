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

:: we disallow local import for non-local packages, if %GOROOT% happens
:: to be under %GOPATH%, then some tests below will fail
set GOPATH=
:: Issue 14340: ignore GOBIN during all.bat.
set GOBIN=

rem TODO avoid rebuild if possible

if x%1==x--no-rebuild goto norebuild
echo ##### Building packages and commands.
go install -a -v std cmd
if errorlevel 1 goto fail
echo.
:norebuild

:: we must unset GOROOT_FINAL before tests, because runtime/debug requires
:: correct access to source code, so if we have GOROOT_FINAL in effect,
:: at least runtime/debug test will fail.
set GOROOT_FINAL=

:: get CGO_ENABLED
go env > env.bat
if errorlevel 1 goto fail
call env.bat
del env.bat
echo.

go tool dist test
if errorlevel 1 goto fail
echo.

goto end

:fail
set GOBUILDFAIL=1

:end
