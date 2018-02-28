:: Copyright 2013 The Go Authors. All rights reserved.
:: Use of this source code is governed by a BSD-style
:: license that can be found in the LICENSE file.

:: race.bash tests the standard library under the race detector.
:: https://golang.org/doc/articles/race_detector.html

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
.\cmd\dist\dist env -w -p >env.bat
if errorlevel 1 goto fail
call env.bat
del env.bat

if %GOHOSTARCH% == amd64 goto continue
echo Race detector is only supported on windows/amd64.
goto fail

:continue
call make.bat --no-banner --no-local
if %GOBUILDFAIL%==1 goto end
echo # go install -race std
go install -race std
if errorlevel 1 goto fail

go tool dist test -race

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

