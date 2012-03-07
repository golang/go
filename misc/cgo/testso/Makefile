# Copyright 2011 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../../src/Make.inc

TARG=cgosotest

CGO_DEPS+=libcgoso.so
CGO_LDFLAGS+=-lcgoso -L.
CLEANFILES+=out libcgoso.so
CGOFILES=\
	cgoso.go\

include ../../../src/Make.pkg

libcgoso.so: cgoso_c.c
	gcc cgoso_c.c -fPIC -o $@ $(_CGO_CFLAGS_$(GOARCH)) $(_CGO_LDFLAGS_$(GOOS))

out: install main.go
	$(GC) $(GCFLAGS) $(GCIMPORTS) main.go
	$(LD) -o $@ main.$O
