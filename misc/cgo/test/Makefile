# Copyright 2011 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../../src/Make.inc

TARG=runtime/cgotest

CGOFILES=\
	align.go\
	basic.go\
	callback.go\
	env.go\
	exports.go\
	helpers.go\
	issue1222.go\
	issue1328.go\
	issue1560.go\
	issue2462.go\
	duplicate_symbol.go\

CGO_OFILES=\
	callback_c.o\

OFILES=\
	runtime.$O\

ifeq ($(GOOS),windows)
GCCVERSION=$(shell gcc -dumpversion)
ifeq ($(GOARCH),386)
GCCLIBDIR=/mingw/lib/gcc/mingw32/$(GCCVERSION)
CHKSTK=_chkstk.o
else
GCCLIBDIR=/mingw/lib/gcc/x86_64-w64-mingw32/$(GCCVERSION)
CHKSTK=_chkstk_ms.o
endif

CGOFILES+=sleep_windows.go
CGO_OFILES+=$(CHKSTK)

$(CHKSTK):
	ar -x "$(GCCLIBDIR)/libgcc.a" $@
endif

include ../../../src/Make.pkg
