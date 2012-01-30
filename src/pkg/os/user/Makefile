# Copyright 2011 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../../Make.inc

TARG=os/user
GOFILES=\
	user.go\

GOFILES_windows=\
	lookup_windows.go

ifeq ($(CGO_ENABLED),1)
CGOFILES_linux=\
	lookup_unix.go
CGOFILES_freebsd=\
	lookup_unix.go
CGOFILES_darwin=\
	lookup_unix.go
endif

ifneq ($(CGOFILES_$(GOOS)),)
CGOFILES+=$(CGOFILES_$(GOOS))
else
ifneq ($(GOOS),windows)
GOFILES+=lookup_stubs.go
endif
endif

GOFILES+=$(GOFILES_$(GOOS))

include ../../../Make.pkg
