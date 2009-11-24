# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../Make.conf

TARG=\
	8c\

HFILES=\
	gc.h\
	../8l/8.out.h\
	../cc/cc.h\

OFILES=\
	cgen.$O\
	cgen64.$O\
	div.$O\
	list.$O\
	machcap.$O\
	mul.$O\
	pgen.$O\
	pswt.$O\
	peep.$O\
	reg.$O\
	sgen.$O\
	swt.$O\
	txt.$O\
	../8l/enam.$O\

LIB=\
	../cc/cc.a$O

$(TARG): $(OFILES) $(LIB)
	$(LD) -o $(TARG) -L"$(GOROOT)"/lib $(OFILES) $(LIB) -lm -lbio -l9

$(OFILES): $(HFILES)

clean:
	rm -f *.$O $(TARG) *.8 8.out a.out

install: $(TARG)
	cp $(TARG) "$(GOBIN)"/$(TARG)

%.$O: ../cc/%.c
	$(CC) $(CFLAGS) -c -I. -o $@ ../cc/$*.c
