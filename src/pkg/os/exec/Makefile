# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../../Make.inc

TARG=os/exec
GOFILES=\
	exec.go\

GOFILES_freebsd=\
	lp_unix.go\

GOFILES_darwin=\
	lp_unix.go\

GOFILES_linux=\
	lp_unix.go\

GOFILES_openbsd=\
	lp_unix.go\

GOFILES_windows=\
	lp_windows.go\

GOFILES_plan9=\
	lp_plan9.go\

GOFILES+=$(GOFILES_$(GOOS))

include ../../../Make.pkg
