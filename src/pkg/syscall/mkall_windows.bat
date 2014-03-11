:: Copyright 2013 The Go Authors. All rights reserved.
:: Use of this source code is governed by a BSD-style
:: license that can be found in the LICENSE file.
@echo off

if exist mkall.sh goto dirok
echo mkall_windows.bat must be run from src\pkg\syscall directory
goto :end
:dirok

if "%1"=="386" goto :paramok
if "%1"=="amd64" goto :paramok
echo parameters must be 386 or amd64
goto :end
:paramok

go build mksyscall_windows.go
.\mksyscall_windows syscall_windows.go security_windows.go syscall_windows_%1.go |gofmt >zsyscall_windows_%1.go
del mksyscall_windows.exe

:end