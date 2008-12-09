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
	const.$O\
	mparith1.$O\
	mparith2.$O\
	mparith3.$O\
	sysimport.$O\
	compat.$O\

$(LIB): $(OFILES)
	ar rsc $(LIB) $(OFILES)

$(OFILES): $(HFILES)

y.tab.h: $(YFILES)
	yacc $(YFLAGS) $(YFILES)

y.tab.c: y.tab.h
	test -f y.tab.c && touch y.tab.c

sysimport.c:	sys.go unsafe.go mksys.c
	gcc -o mksys mksys.c
	6g sys.go
	6g unsafe.go
	./mksys sys >_sysimport.c &&\
		./mksys unsafe >>_sysimport.c &&\
		mv _sysimport.c sysimport.c

clean:
	rm -f $(OFILES) *.6 enam.c 6.out a.out y.tab.h y.tab.c $(LIB) _sysimport.c

install: $(LIB)
