# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../Make.conf

TARG=\
	6g

HFILES=\
	../gc/go.h\
	../6l/6.out.h\
	gg.h\
	opt.h\

OFILES=\
	list.$O\
	obj.$O\
	align.$O\
	gen.$O\
	cgen.$O\
	gsubr.$O\
	peep.$O\
	reg.$O\
	../6l/enam.$O\

LIB=\
	../gc/gc.a$O

$(TARG): $(OFILES) $(LIB)
	$(LD) -o $(TARG) -L$(GOROOT)/lib $(OFILES) $(LIB) -lbio -l9 -lm

$(OFILES): $(HFILES)

clean:
	rm -f $(OFILES) $(TARG) *.6 enam.c 6.out a.out

install: $(TARG)
	cp $(TARG) $(BIN)/$(TARG)
