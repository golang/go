:: Copyright 2013 The Go Authors. All rights reserved.
:: Use of this source code is governed by a BSD-style
:: license that can be found in the LICENSE file.

:: race.bash tests the standard library under the race detector.
:: https://golang.org/doc/articles/race_detector.html

@echo off

setlocal

if not exist make.bat (
    echo race.bat must be run from go\src
    exit /b 1
)

if not "%PROCESSOR_ARCHITECTURE%"=="AMD64" (
    echo Race detector is only supported on windows/amd64.
    exit /b 1
)

call .\make.bat --no-banner || exit /b 1
go install -race std || exit /b 1
go tool dist test -race || exit /b 1
