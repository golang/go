# Copyright 2010 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../../Make.inc

TARG=crypto/rand

GOFILES=\
	rand.go\
	util.go\

GOFILES_freebsd=\
	rand_unix.go\

GOFILES_darwin=\
	rand_unix.go\

GOFILES_linux=\
	rand_unix.go\

GOFILES_windows=\
	rand_windows.go\

GOFILES+=$(GOFILES_$(GOOS))

include ../../../Make.pkg
