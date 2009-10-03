# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../Make.conf

LIB=\
	cc.a$O\

HFILES=\
	cc.h\
	y.tab.h\

YFILES=\
	cc.y\

OFILES=\
	y.tab.$O\
	lex.$O\
	mac.$O\
	dcl.$O\
	acid.$O\
	pickle.$O\
	bits.$O\
	com.$O\
	scon.$O\
	funct.$O\
	sub.$O\
	com64.$O\
	dpchk.$O\
	omachcap.$O\

$(LIB): $(OFILES)
	ar rsc $(LIB) $(OFILES)

$(OFILES): $(HFILES)

y.tab.h: $(YFILES)
	bison -y $(YFLAGS) $(YFILES)

y.tab.c: y.tab.h
	test -f y.tab.c && touch y.tab.c

clean:
	rm -f *.$O *.6 enam.c 6.out a.out y.tab.h y.tab.c $(LIB)

install: $(LIB)
