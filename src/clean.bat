:: Copyright 2012 The Go Authors. All rights reserved.
:: Use of this source code is governed by a BSD-style
:: license that can be found in the LICENSE file.

@echo off

setlocal

go tool dist env -w -p >env.bat || exit /b 1
call .\env.bat
del env.bat
echo.

if not exist %GOTOOLDIR%\dist.exe (
    echo cannot find %GOTOOLDIR%\dist.exe; nothing to clean
    exit /b 1
)

"%GOBIN%\go" clean -i std
"%GOBIN%\go" tool dist clean
"%GOBIN%\go" clean -i cmd
