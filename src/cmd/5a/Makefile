# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../Make.conf

TARG=\
	5a\

HFILES=\
	a.h\
	y.tab.h\
	../5l/5.out.h\
	compat.h\

OFILES=\
	y.tab.$O\
	lex.$O\
	compat.$O\
#	../5l/enam.$O\

YFILES=\
	a.y\

$(TARG): $(OFILES)
	$(LD) -o $(TARG) -L$(GOROOT)/lib $(OFILES) -lbio -l9

$(OFILES): $(HFILES)

lex.$O:	../cc/macbody ../cc/lexbody

y.tab.h: $(YFILES)
	bison -y $(YFLAGS) $(YFILES)

y.tab.c: y.tab.h
	test -f y.tab.c && touch y.tab.c

clean:
	rm -f $(OFILES) $(TARG) *.5 enam.c 5.out a.out y.tab.h y.tab.c

install: $(TARG)
	cp $(TARG) $(BIN)/$(TARG)
