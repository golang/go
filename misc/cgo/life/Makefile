# Copyright 2010 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../../src/Make.inc

TARG=life

CGOFILES=\
	life.go\

CGO_OFILES=\
	c-life.o\

CLEANFILES+=life

include ../../../src/Make.pkg

life: install main.go
	$(GC) main.go
	$(LD) -o $@ main.$O
