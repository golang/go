:: Copyright 2012 The Go Authors. All rights reserved.
:: Use of this source code is governed by a BSD-style
:: license that can be found in the LICENSE file.

@echo off

if not exist ..\bin\go.exe (
    echo Must run run.bat from Go src directory after installing cmd/go.
    exit /b 1
)

setlocal

set GOENV=off
..\bin\go tool dist env > env.bat || exit /b 1
call .\env.bat
del env.bat

set GOPATH=c:\nonexist-gopath
..\bin\go tool dist test --rebuild %* || exit /b 1
