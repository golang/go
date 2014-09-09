:: Copyright 2013 The Go Authors. All rights reserved.
:: Use of this source code is governed by a BSD-style
:: license that can be found in the LICENSE file.
@echo off

if exist mkall.sh goto dirok
echo mkall_windows.bat must be run from src\syscall directory
goto :end
:dirok

go build mksyscall_windows.go
.\mksyscall_windows syscall_windows.go security_windows.go |gofmt >zsyscall_windows.go
del mksyscall_windows.exe

:end
