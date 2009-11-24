# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../Make.conf

TARG=\
	5g

HFILES=\
	../gc/go.h\
	../5l/5.out.h\
	gg.h\
	opt.h\

OFILES=\
	../5l/enam.$O\
	list.$O\
	galign.$O\
	gobj.$O\
	ggen.$O\
	gsubr.$O\
	cgen.$O\
	cgen64.$O

LIB=\
	../gc/gc.a$O

$(TARG): $(OFILES) $(LIB)
	$(LD) -o $(TARG) -L"$(GOROOT)"/lib $(OFILES) $(LIB) -lbio -l9 -lm

$(OFILES): $(HFILES)

clean:
	rm -f *.o $(TARG) *.5 enam.c 5.out a.out

install: $(TARG)
	cp $(TARG) "$(GOBIN)"/$(TARG)
