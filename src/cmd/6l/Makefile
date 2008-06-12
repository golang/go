# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

CFLAGS=-I$(GOROOT)/include
BIN=$(HOME)/bin
O=o

TARG=\
	6l\

OFILES=\
	asm.$O\
	obj.$O\
	optab.$O\
	pass.$O\
	span.$O\
	list.$O\
	compat.$O\
	enam.$O\

HFILES=\
	l.h\
	../6l/6.out.h\


$(TARG): $(OFILES)
	cc -o $(TARG) -L$(GOROOT)/lib $(OFILES) -lbio -l9

$(OFILES): $(HFILES)

enam.c: 6.out.h
	sh mkenam

clean:
	rm -f $(OFILES) $(TARG) *.6 enam.c 6.out a.out

install: $(TARG)
	cp $(TARG) $(BIN)/$(TARG)
