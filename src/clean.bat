:: Copyright 2012 The Go Authors. All rights reserved.
:: Use of this source code is governed by a BSD-style
:: license that can be found in the LICENSE file.

@echo off

setlocal

set GOBUILDFAIL=0

go tool dist env -w -p >env.bat
if errorlevel 1 goto fail
call env.bat
del env.bat
echo.

if exist %GOTOOLDIR%\dist.exe goto distok
echo cannot find %GOTOOLDIR%\dist; nothing to clean
goto fail
:distok

"%GOBIN%\go" clean -i std
"%GOBIN%\go" tool dist clean
"%GOBIN%\go" clean -i cmd

goto end

:fail
set GOBUILDFAIL=1

:end
if x%GOBUILDEXIT%==x1 exit %GOBUILDFAIL%
