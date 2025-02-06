:: Copyright 2012 The Go Authors. All rights reserved.
:: Use of this source code is governed by a BSD-style
:: license that can be found in the LICENSE file.

@echo off

if exist ..\bin\go.exe goto ok
echo Must run run.bat from Go src directory after installing cmd/go.
goto fail
:ok

setlocal

set GOENV=off
..\bin\go tool dist env > env.bat || goto fail
call .\env.bat
del env.bat

set GOPATH=c:\nonexist-gopath
..\bin\go tool dist test --rebuild %* || goto fail
goto :eof

:fail
exit /b 1
