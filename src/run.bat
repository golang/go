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

rem TODO avoid rebuild if possible

if x%1==x--no-rebuild goto norebuild
echo # Building packages and commands.
go install -a -v std
if errorlevel 1 goto fail
echo.
:norebuild

echo # Testing packages.
go test std -short -timeout=120s
if errorlevel 1 goto fail
echo.

echo # runtime -cpu=1,2,4
go test runtime -short -timeout=120s -cpu=1,2,4
if errorlevel 1 goto fail
echo.

echo # sync -cpu=10
go test sync -short -timeout=120s -cpu=10
if errorlevel 1 goto fail
echo.

:: TODO: The other tests in run.bash, especially $GOROOT/test/run.

echo ALL TESTS PASSED
goto end

:fail
set GOBUILDFAIL=1

:end
