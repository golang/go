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

rem TODO avoid rebuild if possible

if x%1==x--no-rebuild goto norebuild
echo # Building packages and commands.
go install -a -v std
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

echo # Testing packages.
go test std -short -timeout=120s
if errorlevel 1 goto fail
echo.

set OLDGOMAXPROCS=%GOMAXPROCS%

:: We set GOMAXPROCS=2 in addition to -cpu=1,2,4 in order to test runtime bootstrap code,
:: creation of first goroutines and first garbage collections in the parallel setting.
echo # GOMAXPROCS=2 runtime -cpu=1,2,4
set GOMAXPROCS=2
go test runtime -short -timeout=300s -cpu=1,2,4
if errorlevel 1 goto fail
echo.

set GOMAXPROCS=%OLDGOMAXPROCS%
set OLDGOMAXPROCS=

echo # sync -cpu=10
go test sync -short -timeout=120s -cpu=10
if errorlevel 1 goto fail
echo.

:: Race detector only supported on Linux and OS X,
:: and only on amd64, and only when cgo is enabled.
if not "%GOHOSTOS%-%GOOS%-%GOARCH%-%CGO_ENABLED%" == "windows-windows-amd64-1" goto norace
echo # Testing race detector.
go test -race -i runtime/race flag
if errorlevel 1 goto fail
go test -race -run=Output runtime/race
if errorlevel 1 goto fail
go test -race -short flag
if errorlevel 1 goto fail
echo.
:norace

echo # ..\test\bench\go1
go test ..\test\bench\go1
if errorlevel 1 goto fail
echo.

:: cgo tests
if x%CGO_ENABLED% == x0 goto nocgo
echo # ..\misc\cgo\life
go run "%GOROOT%\test\run.go" - ..\misc\cgo\life
if errorlevel 1 goto fail
echo.

echo # ..\misc\cgo\stdio
go run "%GOROOT%\test\run.go" - ..\misc\cgo\stdio
if errorlevel 1 goto fail
echo.

:: cgo tests inspect the traceback for runtime functions
set OLDGOTRACEBACK=%GOTRACEBACK%
set GOTRACEBACK=2

echo # ..\misc\cgo\test
go test ..\misc\cgo\test
if errorlevel 1 goto fail
echo.

set GOTRACEBACK=%OLDGOTRACEBACK%
set OLDGOTRACEBACK=

echo # ..\misc\cgo\testso
cd ..\misc\cgo\testso
set FAIL=0
call test.bat
cd ..\..\..\src
if %FAIL%==1 goto fail
echo.
:nocgo

echo # ..\doc\progs
go run "%GOROOT%\test\run.go" - ..\doc\progs
if errorlevel 1 goto fail
echo.

:: TODO: The other tests in run.bash.


set OLDGOMAXPROCS=%GOMAXPROCS%

echo # ..\test
cd ..\test
set FAIL=0
set GOMAXPROCS=
go run run.go
if errorlevel 1 set FAIL=1
cd ..\src
echo.
if %FAIL%==1 goto fail

set GOMAXPROCS=%OLDGOMAXPROCS%
set OLDGOMAXPROCS=

:: echo # Checking API compatibility.
go run "%GOROOT%\src\cmd\api\run.go"
if errorlevel 1 goto fail
echo.

echo ALL TESTS PASSED
goto end

:fail
set GOBUILDFAIL=1

:end
