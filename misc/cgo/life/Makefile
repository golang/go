# Copyright 2010 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../../src/Make.$(GOARCH)

TARG=life

CGOFILES=\
	life.go

LDPATH_freebsd=-Wl,-R,`pwd`
LDPATH_linux=-Wl,-R,`pwd`
LDPATH_darwin=

CGO_LDFLAGS=_cgo_export.o c-life.so $(LDPATH_$(GOOS))
CGO_DEPS=_cgo_export.o c-life.so

CLEANFILES += life

include ../../../src/Make.pkg

c-life.o: c-life.c _cgo_export.h
	gcc $(_CGO_CFLAGS_$(GOARCH)) -g -c -fPIC $(CFLAGS) c-life.c

c-life.so: c-life.o
	gcc $(_CGO_CFLAGS_$(GOARCH)) -o $@ c-life.o $(_CGO_LDFLAGS_$(GOOS))

life: install main.go
	$(GC) main.go
	$(LD) -o $@ main.$O
