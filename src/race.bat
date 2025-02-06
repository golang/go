:: Copyright 2013 The Go Authors. All rights reserved.
:: Use of this source code is governed by a BSD-style
:: license that can be found in the LICENSE file.

:: race.bash tests the standard library under the race detector.
:: https://golang.org/doc/articles/race_detector.html

@echo off

setlocal

if exist make.bat goto ok
echo race.bat must be run from go\src
exit /b 1
:ok

set GOROOT=%CD%\..
call .\make.bat --dist-tool >NUL || goto fail
.\cmd\dist\dist.exe env -w -p >env.bat || goto fail
call .\env.bat
del env.bat

if %GOHOSTARCH% == amd64 goto continue
echo Race detector is only supported on windows/amd64.
goto fail

:continue
call .\make.bat --no-banner || goto fail
echo # go install -race std
go install -race std || goto fail
go tool dist test -race || goto fail

echo All tests passed.
goto :eof

:fail
echo Fail.
exit /b 1
