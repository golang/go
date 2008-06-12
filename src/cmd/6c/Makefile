# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

CFLAGS=-I/home/r/plan9/include -I$(GOROOT)/include
BIN=$(HOME)/bin
O=o

TARG=\
	6c\

HFILES=\
	gc.h\
	../6l/6.out.h\
	../cc/cc.h\

OFILES=\
	cgen.$O\
	list.$O\
	sgen.$O\
	swt.$O\
	txt.$O\
	pgen.$O\
	pswt.$O\
	div.$O\
	mul.$O\
	reg.$O\
	peep.$O\
	../6l/enam.$O\

LIB=\
	../cc/cc.a$O

$(TARG): $(OFILES)
	cc -o $(TARG) -L/home/r/plan9/lib $(OFILES) $(LIB) -lbio -l9

$(OFILES): $(HFILES)

clean:
	rm -f $(OFILES) $(TARG) *.6 enam.c 6.out a.out

install: $(TARG)
	cp $(TARG) $(BIN)/$(TARG)
