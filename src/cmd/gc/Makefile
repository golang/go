# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../Make.inc
O:=$(HOST_O)

LIB=gc.a$O

HFILES=\
	go.h\
	y.tab.h\
	md5.h\

OFILES=\
	align.$O\
	bits.$O\
	builtin.$O\
	closure.$O\
	const.$O\
	dcl.$O\
	esc.$O\
	export.$O\
	fmt.$O\
	gen.$O\
	init.$O\
	inl.$O\
	lex.$O\
	md5.$O\
	mparith1.$O\
	mparith2.$O\
	mparith3.$O\
	obj.$O\
	order.$O\
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

HOST_CFLAGS+=-DGOEXPERIMENT='"$(GOEXPERIMENT)"' 

NOINSTALL=1
include ../../Make.clib

install: $(LIB)

subr.$O: yerr.h

fmt.$O: opnames.h

opnames.h:	mkopnames go.h
	./mkopnames go.h >opnames.h

CLEANFILES+=*.[568] [568].out mkbuiltin1 _builtin.c opnames.h

mkbuiltin1: mkbuiltin1.$O
	$(HOST_LD) -o $@ mkbuiltin1.$O -L"$(GOROOT)"/lib -lbio -l9 -lm $(HOST_LDFLAGS)

y.tab.h: go.y
	bison -v -y -d go.y
	# make yystate global, yytname mutable
	cat y.tab.c | sed '/ int yystate;/d; s/int yychar;/int yychar, yystate;/; s/static const char \*const yytname/const char *yytname/; s/char const \*yymsgp/char *yymsgp/' >y1.tab.c
	mv y1.tab.c y.tab.c
	awk -f bisonerrors y.output go.errors >yerr.h

yerr.h: y.tab.h
	test -f yerr.h && touch yerr.h

