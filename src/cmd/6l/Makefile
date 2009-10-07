# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../Make.conf

TARG=\
	6l\

OFILES=\
	asm.$O\
	elf.$O\
	enam.$O\
	go.$O\
	lib.$O\
	list.$O\
	macho.$O\
	obj.$O\
	optab.$O\
	pass.$O\
	span.$O\

HFILES=\
	l.h\
	../6l/6.out.h\
	../ld/lib.h\
	../ld/elf.h\
	../ld/macho.h\

$(TARG): $(OFILES)
	$(LD) -o $(TARG) -L$(GOROOT)/lib $(OFILES) -lbio -l9

$(OFILES): $(HFILES)

enam.c: 6.out.h
	sh mkenam

clean:
	rm -f *.$O $(TARG) *.6 enam.c 6.out a.out

install: $(TARG)
	cp $(TARG) $(GOBIN)/$(TARG)

%.$O: ../ld/%.c
	$(CC) $(CFLAGS) -c -I. ../ld/$*.c
