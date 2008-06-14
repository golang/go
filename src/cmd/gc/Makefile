# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

YFLAGS=-d
CFLAGS=-I$(GOROOT)/include
BIN=$(HOME)/bin
O=o

LIB=\
	gc.a$O\

HFILES=\
	go.h\
	y.tab.h\

YFILES=\
	go.y\

OFILES=\
	y.tab.$O\
	lex.$O\
	subr.$O\
	dcl.$O\
	export.$O\
	walk.$O\
	const.$O\
	mpatof.$O\
	sysimport.$O\
	compat.$O\

$(LIB): $(OFILES)
	ar rsc $(LIB) $(OFILES)

$(OFILES): $(HFILES)

y.tab.c: $(YFILES)
	yacc $(YFLAGS) $(YFILES)

sysimport.c:	sys.go mksys.bash
	bash mksys.bash

clean:
	rm -f $(OFILES) *.6 enam.c 6.out a.out y.tab.h y.tab.c $(LIB)

install: $(LIB)
