# Copyright 2009 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

CFLAGS=-I$(GOOS) -I$(GOOS)/$(GOARCH) -wF
O=6
CC=$(O)c
AS=$(O)a

LIB=lib.a

OFILES=\
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
	mheapmap64.$O\
	msize.$O\
	print.$O\
	proc.$O\
	rune.$O\
	runtime.$O\
	rt0.$O\
	sema.$O\
	sema_go.$O\
	signal.$O\
	string.$O\
	symtab.$O\
	sys.$O\
	thread.$O\
	traceback.$O\

HFILES=\
	runtime.h\
	hashmap.h\
	malloc.h\
	$(GOOS)/os.h\
	$(GOOS)/$(GOARCH)/defs.h\

install: $(LIB) runtime.acid
	cp $(LIB) $(GOROOT)/lib/lib_$(GOARCH)_$(GOOS).a
	cp runtime.acid $(GOROOT)/acid/runtime.acid

$(LIB): $(OFILES)
	$(O)ar rc $(LIB) $(OFILES)

$(OFILES): $(HFILES)

nuke:
	rm -f *.$(O) *.a $(GOROOT)/lib/$(LIB)

clean:
	rm -f *.$(O) *.a runtime.acid cgo2c

%.$O:	%.c
	$(CC) $(CFLAGS) $<

%.$O:	$(GOARCH)/%.c
	$(CC) $(CFLAGS) $<

%.$O:	$(GOOS)/%.c
	$(CC) $(CFLAGS) $<

%.$O:	$(GOOS)/$(GOARCH)/%.c
	$(CC) $(CFLAGS) $<

%.$O:	$(GOARCH)/%.s
	$(AS) $<

%.$O:	$(GOOS)/$(GOARCH)/%.s
	$(AS) $<

cgo2c: cgo2c.c
	quietgcc -o $@ $<

%.c:	%.cgo cgo2c
	./cgo2c $< > $@.tmp
	mv -f $@.tmp $@

runtime.acid: runtime.h proc.c
	$(CC) -a proc.c >runtime.acid

