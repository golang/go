# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../Make.inc
O:=$(HOST_O)

LIB=gc.a

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
	esc.$O\
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

HOST_CFLAGS+=-DGOEXPERIMENT='"$(GOEXPERIMENT)"' 

NOINSTALL=1
include ../../Make.clib

install: $(LIB)

y1.tab.c: y.tab.c	# make yystate global, yytname mutable
	cat y.tab.c | sed '/ int yystate;/d; s/int yychar;/int yychar, yystate;/; s/static const char \*const yytname/const char *yytname/; s/char const \*yymsgp/char *yymsgp/' >y1.tab.c

yerr.h: bisonerrors go.errors y.tab.h	# y.tab.h rule generates y.output too
	awk -f bisonerrors y.output go.errors >yerr.h

subr.$O: yerr.h

builtin.c: builtin.c.boot
	cp builtin.c.boot builtin.c

subr.$O: opnames.h

opnames.h:	mkopnames go.h
	./mkopnames go.h >opnames.h

CLEANFILES+=*.[568] [568].out y1.tab.c yerr.h mkbuiltin1 builtin.c _builtin.c opnames.h

mkbuiltin1: mkbuiltin1.$O
	$(HOST_LD) -o $@ mkbuiltin1.$O -L"$(GOROOT)"/lib -lbio -l9 -lm $(HOST_LDFLAGS)

