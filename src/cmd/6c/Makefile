# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../Make.conf

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

$(TARG): $(OFILES) $(LIB)
	$(LD) -o $(TARG) -L"$(GOROOT)"/lib $(OFILES) $(LIB) -lbio -l9

$(OFILES): $(HFILES)

clean:
	rm -f *.$O $(TARG) *.6 enam.c 6.out a.out

install: $(TARG)
	cp $(TARG) "$(GOBIN)"/$(TARG)

%.$O: ../cc/%.c
	$(CC) $(CFLAGS) -c -I. -o $@ ../cc/$*.c
