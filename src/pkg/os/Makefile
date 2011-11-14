# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../Make.inc

TARG=os
GOFILES=\
	error.go\
	env.go\
	exec.go\
	file.go\
	getwd.go\
	path.go\
	proc.go\
	stat_$(GOOS).go\
	time.go\
	types.go\

GOFILES_freebsd=\
	dir_unix.go\
	error_posix.go\
	file_posix.go\
	file_unix.go\
	path_unix.go\
	sys_bsd.go\
	exec_posix.go\
	exec_unix.go\
	signal_unix.go\

GOFILES_darwin=\
	dir_unix.go\
	error_posix.go\
	file_posix.go\
	file_unix.go\
	path_unix.go\
	sys_bsd.go\
	exec_posix.go\
	exec_unix.go\
	signal_unix.go\

GOFILES_linux=\
	dir_unix.go\
	error_posix.go\
	file_posix.go\
	file_unix.go\
	path_unix.go\
	sys_linux.go\
	exec_posix.go\
	exec_unix.go\
	signal_unix.go\

GOFILES_openbsd=\
	dir_unix.go\
	error_posix.go\
	file_posix.go\
	file_unix.go\
	path_unix.go\
	sys_bsd.go\
	exec_posix.go\
	exec_unix.go\
	signal_unix.go\

GOFILES_windows=\
	dir_windows.go\
	error_posix.go\
	file_posix.go\
	file_windows.go\
	path_windows.go\
	sys_windows.go\
	exec_posix.go\
	exec_windows.go\
	signal_windows.go\

GOFILES_plan9=\
	dir_plan9.go\
	error_plan9.go\
	file_plan9.go\
	path_plan9.go\
	sys_plan9.go\
	exec_plan9.go\
	str.go\

GOFILES+=$(GOFILES_$(GOOS))

CLEANFILES+=signal_unix.go signal_windows.go

include ../../Make.pkg

signal_unix.go: ../syscall/zerrors_$(GOOS)_$(GOARCH).go
	./mkunixsignals.sh $< > $@ || rm -f $@

signal_windows.go: ../syscall/ztypes_$(GOOS).go
	./mkunixsignals.sh $< > $@ || rm -f $@
