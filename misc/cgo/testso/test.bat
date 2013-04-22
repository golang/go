:: Copyright 2013 The Go Authors.  All rights reserved.
:: Use of this source code is governed by a BSD-style
:: license that can be found in the LICENSE file.

@echo off

gcc -c cgoso_c.c
gcc -shared -o libcgosotest.dll cgoso_c.o
if not exist libcgosotest.dll goto fail
go build main.go
if not exist main.exe goto fail
main.exe
goto :end

:fail
set FAIL=1
:end
del /F cgoso_c.o libcgosotest.dll main.exe 2>NUL
