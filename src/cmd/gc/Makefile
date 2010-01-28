# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../Make.conf

LIB=\
	gc.a$O\

HFILES=\
	go.h\
	y.tab.h\
	md5.h\

YFILES=\
	go.y\

OFILES=\
	align.$O\
	bits.$O\
	builtin.$O\
	closure.$O\
	const.$O\
	dcl.$O\
	export.$O\
	gen.$O\
	init.$O\
	lex.$O\
	md5.$O\
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
	y1.tab.$O\

$(LIB): $(OFILES)
	ar rsc $(LIB) $(OFILES)

$(OFILES): $(HFILES)

y.tab.h: $(YFILES)
	LANG=C bison -v -y $(YFLAGS) $(YFILES)

y.tab.c: y.tab.h
	test -f y.tab.c && touch y.tab.c

y1.tab.c: y.tab.c	# make yystate global, yytname mutable
	cat y.tab.c | sed '/ int yystate;/d; s/int yychar;/int yychar, yystate;/; s/static const char \*const yytname/const char *yytname/' >y1.tab.c

yerr.h: bisonerrors go.errors y.tab.h	# y.tab.h rule generates y.output too
	awk -f bisonerrors y.output go.errors >yerr.h

subr.$O: yerr.h

builtin.c: builtin.c.boot
	cp builtin.c.boot builtin.c

subr.$O: opnames.h

opnames.h:	mkopnames go.h
	./mkopnames go.h >opnames.h

clean:
	rm -f *.[568o] enam.c [568].out a.out y.tab.h y.tab.c y1.tab.c y.output yerr.h $(LIB) mkbuiltin1 builtin.c _builtin.c opnames.h

install: $(LIB)
