# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../Make.conf

TARG=\
	5l\

OFILES=\
	asm.$O\
	enam.$O\
	lib.$O\
	list.$O\
	noop.$O\
	obj.$O\
	optab.$O\
	pass.$O\
	thumb.$O\
	span.$O\
	go.$O\

HFILES=\
	l.h\
	../5l/5.out.h\

$(TARG): $(OFILES)
	$(LD) -o $(TARG) -L$(GOROOT)/lib $(OFILES) -lbio -l9

$(OFILES): $(HFILES)

enam.c: 5.out.h
	sh mkenam

clean:
	rm -f *.o $(TARG) *.5 enam.c 5.out a.out

install: $(TARG)
	cp $(TARG) $(GOBIN)/$(TARG)

%.$O: ../ld/%.c
	$(CC) $(CFLAGS) -c -I. ../ld/$*.c
