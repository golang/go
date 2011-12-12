# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../Make.inc

TARG=mime
GOFILES=\
	grammar.go\
	mediatype.go\
	type.go\

GOFILES_freebsd=\
	type_unix.go

GOFILES_darwin=\
	type_unix.go

GOFILES_linux=\
	type_unix.go

GOFILES_netbsd=\
	type_unix.go

GOFILES_openbsd=\
	type_unix.go

GOFILES_plan9=\
	type_unix.go

GOFILES_windows=\
	type_windows.go

GOFILES+=$(GOFILES_$(GOOS))

include ../../Make.pkg
