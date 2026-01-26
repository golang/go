:: Copyright 2012 The Go Authors. All rights reserved.
:: Use of this source code is governed by a BSD-style
:: license that can be found in the LICENSE file.

@echo off

setlocal

if not exist make.bat (
    echo all.bat must be run from go\src
    exit /b 1
)

call .\make.bat --no-banner || exit /b 1
call .\run.bat --no-rebuild || exit /b 1
..\bin\go tool dist banner
