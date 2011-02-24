# Copyright 2011 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

GOOS=windows

include ../../../Make.inc

LD:=$(LD) -Hwindowsgui

TARG=wingui

GOFILES=\
	gui.go\
	winapi.go\
	zwinapi.go\

include ../../../Make.cmd

zwinapi.go: winapi.go
	$(GOROOT)/src/pkg/syscall/mksyscall_windows.sh $< \
		| sed 's/^package.*syscall$$/package main/' \
		| sed '/^import/a \
			import "syscall"' \
		| sed 's/Syscall/syscall.Syscall/' \
		| sed 's/EINVAL/syscall.EINVAL/' \
		| gofmt \
		> $@
