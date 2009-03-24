# Copyright 2009 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

CFLAGS=
O=6
CC=$(O)c
AS=$(O)a

LIB=lib_$(GOARCH)_$(GOOS).a

RT0OFILES=\
	rt0_$(GOARCH)_$(GOOS).$O\

LIBOFILES=\
	rt1_$(GOARCH)_$(GOOS).$O\
	sys_$(GOARCH)_$(GOOS).$O\
	array.$O\
	asm.$O\
	chan.$O\
	closure.$O\
	float.$O\
	float_go.$O\
	hashmap.$O\
	iface.$O\
	malloc.$O\
	malloc_go.$O\
	mcache.$O\
	mcentral.$O\
	mem.$O\
	mfixalloc.$O\
	mgc0.$O\
	mheap.$O\
	msize.$O\
	print.$O\
	proc.$O\
	rune.$O\
	runtime.$O\
	sema.$O\
	sema_go.$O\
	string.$O\
	symtab.$O\
	traceback.$O\

OFILES=$(RT0OFILES) $(LIBOFILES)
OS_H=$(GOARCH)_$(GOOS).h
HFILES=runtime.h hashmap.h malloc.h $(OS_H_)

install: rt0 $(LIB) runtime.acid
	cp $(RT0OFILES) $(GOROOT)/lib
	cp $(LIB) $(GOROOT)/lib
	cp runtime.acid $(GOROOT)/acid/runtime.acid

rt0:	$(RT0OFILES)

$(LIB): $(LIBOFILES)
	$(O)ar rc $(LIB) $(LIBOFILES)

$(OFILES): $(HFILES)

nuke:
	rm -f *.$(O) *.a $(GOROOT)/lib/$(LIB)

clean:
	rm -f *.$(O) *.a runtime.acid cgo2c

%.$O:	%.c
	$(CC) $(CFLAGS) -wF $<

%.$O:	$(GOARCH)/%.c
	$(CC) $(CFLAGS) -wF $<

%.$O:	%.s
	$(AS) $<

%.$O:	$(GOARCH)/%.s
	$(AS) $<

cgo2c: cgo2c.c
	quietgcc -o $@ $<

%.c:	%.cgo cgo2c
	./cgo2c $< > $@.tmp
	mv -f $@.tmp $@

runtime.acid: runtime.h proc.c
	$(CC) -a proc.c >runtime.acid

