# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../Make.conf

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
	swt.$O\
	const.$O\
	mparith1.$O\
	mparith2.$O\
	mparith3.$O\
	builtin.$O\
	compat.$O\
	bits.$O\
	align.$O\

$(LIB): $(OFILES)
	ar rsc $(LIB) $(OFILES)

$(OFILES): $(HFILES)

y.tab.h: $(YFILES)
	bison -y $(YFLAGS) $(YFILES)

y.tab.c: y.tab.h
	test -f y.tab.c && touch y.tab.c

builtin.c:	sys.go unsafe.go mkbuiltin1.c mkbuiltin
	./mkbuiltin >builtin.c || \
	(echo 'mkbuiltin failed; using bootstrap copy of builtin.c'; cp builtin.c.boot builtin.c)

clean:
	rm -f $(OFILES) *.6 enam.c 6.out a.out y.tab.h y.tab.c $(LIB) mkbuiltin1 builtin.c _builtin.c

install: $(LIB)
