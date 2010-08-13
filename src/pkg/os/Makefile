# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../Make.$(GOARCH)

TARG=os
GOFILES=\
	dir_$(GOOS).go\
	error.go\
	exec.go\
	file.go\
	getwd.go\
	path.go\
	proc.go\
	stat_$(GOOS).go\
	time.go\
	types.go\

GOFILES_freebsd=\
	env_unix.go\
	file_unix.go\
	sys_bsd.go\

GOFILES_darwin=\
	env_unix.go\
	file_unix.go\
	sys_bsd.go\

GOFILES_linux=\
	env_unix.go\
	file_unix.go\
	sys_linux.go\

GOFILES_nacl=\
	env_unix.go\
	file_unix.go\
	sys_nacl.go\

GOFILES_windows=\
	env_windows.go\
	file_windows.go\
	sys_windows.go\

GOFILES+=$(GOFILES_$(GOOS))

include ../../Make.pkg
