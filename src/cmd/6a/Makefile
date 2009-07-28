# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../Make.conf

TARG=\
	6a\

HFILES=\
	a.h\
	y.tab.h\
	../6l/6.out.h\

OFILES=\
	y.tab.$O\
	lex.$O\
	../6l/enam.$O\

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
	rm -f $(OFILES) $(TARG) *.6 enam.c 6.out a.out y.tab.h y.tab.c

install: $(TARG)
	cp $(TARG) $(GOBIN)/$(TARG)
