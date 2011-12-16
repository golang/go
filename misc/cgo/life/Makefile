# Copyright 2010 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../../src/Make.inc

TARG=life

CGOFILES=\
	life.go\

CGO_OFILES=\
	c-life.o\

ifeq ($(GOOS),windows)
ifeq ($(GOARCH),amd64)
CGO_OFILES+=\
	lib64_libmingwex_a-wassert.o\
	lib64_libmingw32_a-mingw_helpers.o\

lib64_libmingwex_a-wassert.o:
	ar -x /mingw/x86_64-w64-mingw32/lib/libmingwex.a lib64_libmingwex_a-wassert.o

lib64_libmingw32_a-mingw_helpers.o:
	ar -x /mingw/x86_64-w64-mingw32/lib/libmingw32.a  lib64_libmingw32_a-mingw_helpers.o
endif
endif

CLEANFILES+=life

include ../../../src/Make.pkg

life: install main.go
	$(GC) $(GCFLAGS) $(GCIMPORTS) main.go
	$(LD) -o $@ main.$O
