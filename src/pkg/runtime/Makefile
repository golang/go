# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include $(GOROOT)/src/Make.$(GOARCH)

TARG=runtime

# Set SIZE to 32 or 64.
SIZE_386=32
SIZE_amd64=64
SIZE_arm=32
SIZE=$(SIZE_$(GOARCH))

# Setup CFLAGS.  Add -D_64BIT on 64-bit platforms (sorry).
CFLAGS_64=-D_64BIT
# TODO(kaib): fix register allocation to honor extern register so we
# can enable optimizations again.
CFLAGS_arm=-N
CFLAGS=-I$(GOOS) -I$(GOOS)/$(GOARCH) -wF $(CFLAGS_$(SIZE)) $(CFLAGS_$(GOARCH))

GOFILES=\
	extern.go\
	type.go\

# 386-specific object files
OFILES_386=\
	vlop.$O\
	vlrt.$O\

# arm-specific object files
OFILES_arm=\
	memset.$O\
	vlop.$O\
	vlrt.$O\

OFILES=\
	array.$O\
	asm.$O\
	chan.$O\
	closure.$O\
	float.$O\
	hashmap.$O\
	iface.$O\
	malloc.$O\
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
	reflect.$O\
	rune.$O\
	runtime.$O\
	rt0.$O\
	sema.$O\
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
	$(GOARCH)/asm.h\
	$(GOOS)/os.h\
	$(GOOS)/$(GOARCH)/defs.h\

include $(GOROOT)/src/Make.pkg

clean: clean-local

clean-local:
	rm -f runtime.acid cgo2c */asm.h

$(GOARCH)/asm.h: runtime.acid mkasmh
	./mkasmh >$@.x
	mv -f $@.x $@

cgo2c: cgo2c.c
	quietgcc -o $@ $<

%.c:	%.cgo cgo2c
	./cgo2c $< > $@.tmp
	mv -f $@.tmp $@

%.$O:	$(GOARCH)/%.c
	$(CC) $(CFLAGS) $<

%.$O:	$(GOOS)/%.c
	$(CC) $(CFLAGS) $<

%.$O:	$(GOOS)/$(GOARCH)/%.c
	$(CC) $(CFLAGS) $<

%.$O:	$(GOARCH)/%.s $(GOARCH)/asm.h
	$(AS) $<

%.$O:	$(GOOS)/$(GOARCH)/%.s $(GOARCH)/asm.h
	$(AS) $<


# acid support files

all: runtime.acid

install: $(GOROOT)/acid/runtime.acid

$(GOROOT)/acid/runtime.acid: runtime.acid
	cp runtime.acid $@

runtime.acid: runtime.h proc.c
	$(CC) -a proc.c >runtime.acid

chan.acid: runtime.h chan.c
	$(CC) -a chan.c >chan.acid

