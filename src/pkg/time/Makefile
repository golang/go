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
	zoneinfo_unix.go\

GOFILES_darwin=\
	zoneinfo_unix.go\

GOFILES_linux=\
	zoneinfo_unix.go\

GOFILES_windows=\
	zoneinfo_windows.go\

GOFILES+=$(GOFILES_$(GOOS))

include ../../Make.pkg
