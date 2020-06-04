:: Copyright 2012 The Go Authors. All rights reserved.
:: Use of this source code is governed by a BSD-style
:: license that can be found in the LICENSE file.

@echo off

setlocal

if exist make.bat goto ok
echo all.bat must be run from go\src
:: cannot exit: would kill parent command interpreter
goto end
:ok

set OLDPATH=%PATH%
call make.bat --no-banner --no-local
if %GOBUILDFAIL%==1 goto end
call run.bat --no-rebuild --no-local
if %GOBUILDFAIL%==1 goto end
:: we must restore %PATH% before running "dist banner" so that the latter
:: can get the original %PATH% and give suggestion to add %GOROOT%/bin
:: to %PATH% if necessary.
set PATH=%OLDPATH%
"%GOTOOLDIR%/dist" banner

:end
if x%GOBUILDEXIT%==x1 exit %GOBUILDFAIL%
