# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../../Make.inc

TARG=path/filepath
GOFILES=\
	match.go\
	path.go\

GOFILES_freebsd=\
	path_unix.go

GOFILES_darwin=\
	path_unix.go

GOFILES_linux=\
	path_unix.go

GOFILES_windows=\
	path_unix.go

GOFILES+=$(GOFILES_$(GOOS))

include ../../../Make.pkg
