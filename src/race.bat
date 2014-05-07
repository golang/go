:: Copyright 2013 The Go Authors. All rights reserved.
:: Use of this source code is governed by a BSD-style
:: license that can be found in the LICENSE file.

:: race.bash tests the standard library under the race detector.
:: http://golang.org/doc/articles/race_detector.html

@echo off

setlocal

if exist make.bat goto ok
echo race.bat must be run from go\src
:: cannot exit: would kill parent command interpreter
goto end
:ok

set GOROOT=%CD%\..
call make.bat --dist-tool >NUL
if errorlevel 1 goto fail
.\cmd\dist\dist env -wp >env.bat
if errorlevel 1 goto fail
call env.bat
del env.bat

if %GOHOSTARCH% == amd64 goto continue
echo Race detector is only supported on windows/amd64.
goto fail

:continue
call make.bat --no-banner --no-local
if %GOBUILDFAIL%==1 goto end
:: golang.org/issue/5537 - we must build a race enabled cmd/cgo before trying to use it.
echo # go install -race cmd/cgo
go install -race cmd/cgo
echo # go install -race std
go install -race std
if errorlevel 1 goto fail

:: we must unset GOROOT_FINAL before tests, because runtime/debug requires
:: correct access to source code, so if we have GOROOT_FINAL in effect,
:: at least runtime/debug test will fail.
set GOROOT_FINAL=

echo # go test -race -short std
go test -race -short std
if errorlevel 1 goto fail
echo # go test -race -run=nothingplease -bench=.* -benchtime=.1s -cpu=4 std
go test -race -run=nothingplease -bench=.* -benchtime=.1s -cpu=4 std
if errorlevel 1 goto fail
goto succ

:fail
set GOBUILDFAIL=1
echo Fail.
goto end

:succ
echo All tests passed.

:end
if x%GOBUILDEXIT%==x1 exit %GOBUILDFAIL%

