:: Copyright 2012 The Go Authors. All rights reserved.
:: Use of this source code is governed by a BSD-style
:: license that can be found in the LICENSE file.
@echo off

if exist make.bat goto ok
echo all.bat must be run from go\src
:: cannot exit: would kill parent command interpreter
goto end
:ok

set GOOLDPATH=%PATH%

call make.bat --no-banner
if %GOBUILDFAIL%==1 goto end
call run.bat --no-rebuild
if %GOBUILDFAIL%==1 goto end
go tool dist banner

:end
set PATH=%GOOLDPATH%
if x%GOBUILDEXIT%==x1 exit %GOBUILDFAIL%
