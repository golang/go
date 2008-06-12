# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

CFLAGS=-I/home/r/plan9/include -I$(GOROOT)/include
BIN=$(HOME)/bin
O=o

TARG=\
	6g

HFILES=\
	../gc/go.h\
	../6l/6.out.h\
	gg.h\

OFILES=\
	list.$O\
	align.$O\
	gen.$O\
	cgen.$O\
	gsubr.$O\
	obj.$O\
	../6l/enam.$O\

LIB=\
	../gc/gc.a$O

$(TARG): $(OFILES) $(LIB)
	cc -o $(TARG) -L/home/r/plan9/lib $(OFILES) $(LIB) -lbio -l9

$(OFILES): $(HFILES)

clean:
	rm -f $(OFILES) $(TARG) *.6 enam.c 6.out a.out

install: $(TARG)
	cp $(TARG) $(BIN)/$(TARG)
