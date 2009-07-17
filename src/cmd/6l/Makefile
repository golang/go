# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../Make.conf

TARG=\
	6l\

OFILES=\
	asm.$O\
	compat.$O\
	elf64.$O\
	enam.$O\
	go.$O\
	list.$O\
	obj.$O\
	optab.$O\
	pass.$O\
	span.$O\

HFILES=\
	l.h\
	../6l/6.out.h\
	../ld/elf64.h\


$(TARG): $(OFILES)
	$(LD) -o $(TARG) -L$(GOROOT)/lib $(OFILES) -lbio -l9

$(OFILES): $(HFILES)

enam.c: 6.out.h
	sh mkenam

clean:
	rm -f $(OFILES) $(TARG) *.6 enam.c 6.out a.out

install: $(TARG)
	cp $(TARG) $(BIN)/$(TARG)

go.o: ../ld/go.c

elf64.o: ../ld/elf64.c
