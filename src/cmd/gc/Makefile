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
	align.$O\
	bits.$O\
	builtin.$O\
	compat.$O\
	const.$O\
	dcl.$O\
	export.$O\
	gen.$O\
	init.$O\
	lex.$O\
	mparith1.$O\
	mparith2.$O\
	mparith3.$O\
	obj.$O\
	print.$O\
	range.$O\
	reflect.$O\
	select.$O\
	sinit.$O\
	subr.$O\
	swt.$O\
	typecheck.$O\
	unsafe.$O\
	walk.$O\
	y.tab.$O\

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

subr.$O: opnames.h

opnames.h:	mkopnames go.h
	./mkopnames go.h >opnames.h

clean:
	rm -f $(OFILES) *.6 enam.c 6.out a.out y.tab.h y.tab.c $(LIB) mkbuiltin1 builtin.c _builtin.c

install: $(LIB)
