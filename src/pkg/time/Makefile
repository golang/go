# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../Make.inc

TARG=time
GOFILES=\
	format.go\
	sleep.go\
	sys.go\
	tick.go\
	time.go\

GOFILES_freebsd=\
	sys_posix.go\
	zoneinfo_posix.go\
	zoneinfo_unix.go\

GOFILES_darwin=\
	sys_posix.go\
	zoneinfo_posix.go\
	zoneinfo_unix.go\

GOFILES_linux=\
	sys_posix.go\
	zoneinfo_posix.go\
	zoneinfo_unix.go\

GOFILES_windows=\
	sys_posix.go\
	zoneinfo_windows.go\

GOFILES_plan9=\
	sys_plan9.go\
	zoneinfo_posix.go\
	zoneinfo_plan9.go\

GOFILES+=$(GOFILES_$(GOOS))

include ../../Make.pkg
