# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Set SIZE to 32 or 64.
SIZE_386=32
SIZE_amd64=64
SIZE_arm=32
SIZE=$(SIZE_$(GOARCH))

# Setup CFLAGS.  Add -D_64BIT on 64-bit platforms (sorry).
CFLAGS_64=-D_64BIT
CFLAGS=-I$(GOOS) -I$(GOOS)/$(GOARCH) -wF $(CFLAGS_$(SIZE))

# Set O to right letter.
O_386=8
O_amd64=6
O_arm=5
O=$(O_$(GOARCH))

# Tools
CC=$(O)c
GC=$(O)g
AS=$(O)a
AR=gopack

LIB=runtime.a

# 386-specific object files
OFILES_386=\
	vlop.$O\
	vlrt.$O\

OFILES=\
	array.$O\
	asm.$O\
	chan.$O\
	closure.$O\
	extern.$O\
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
	mheapmap$(SIZE).$O\
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
	$(OFILES_$(GOARCH))\

HFILES=\
	runtime.h\
	hashmap.h\
	malloc.h\
	$(GOOS)/os.h\
	$(GOOS)/$(GOARCH)/defs.h\

install: $(LIB) runtime.acid
	test -d $(GOROOT)/pkg && mkdir -p $(GOROOT)/pkg/$(GOOS)_$(GOARCH)
	cp $(LIB) $(GOROOT)/pkg/$(GOOS)_$(GOARCH)/$(LIB)
	cp runtime.acid $(GOROOT)/acid/runtime.acid

$(LIB): $(OFILES)
	$(AR) grc $(LIB) $(OFILES)

$(OFILES): $(HFILES)

nuke:
	rm -f *.[568] *.a $(GOROOT)/lib/$(LIB)

clean:
	rm -f *.[568] *.a runtime.acid cgo2c

%.$O:	%.go
	$(GC) $<

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

chan.acid: runtime.h chan.c
	$(CC) -a chan.c >chan.acid

