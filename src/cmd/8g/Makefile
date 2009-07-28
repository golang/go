# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../Make.conf

TARG=\
	8g

HFILES=\
	../gc/go.h\
	../8l/8.out.h\
	gg.h\
#	opt.h\

OFILES=\
	../8l/enam.$O\
	list.$O\
	galign.$O\
	gobj.$O\
	ggen.$O\
	gsubr.$O\
	cgen.$O\
	cgen64.$O\
#	peep.$O\
#	reg.$O\

LIB=\
	../gc/gc.a$O

$(TARG): $(OFILES) $(LIB)
	$(LD) -o $(TARG) -L$(GOROOT)/lib $(OFILES) $(LIB) -lbio -l9 -lm

$(OFILES): $(HFILES)

clean:
	rm -f $(OFILES) $(TARG) *.8 enam.c 8.out a.out

install: $(TARG)
	cp $(TARG) $(GOBIN)/$(TARG)
