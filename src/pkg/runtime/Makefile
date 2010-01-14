# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../Make.$(GOARCH)

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
CFLAGS_mingw=-D__MINGW__
CFLAGS=-I$(GOOS) -I$(GOOS)/$(GOARCH) -wF $(CFLAGS_$(SIZE)) $(CFLAGS_$(GOARCH)) $(CFLAGS_$(GOOS))

GOFILES=\
	extern.go\
	type.go\

GOFILES_pchw=\
	pchw/io.go\

# 386-specific object files
OFILES_386=\
	vlop.$O\
	vlrt.$O\

GOARM?=6

# arm-specific object files
OFILES_arm=\
	cas$(GOARM).$O\
	memset.$O\
	softfloat.$O\
	vlop.$O\
	vlrt.$O\

OFILES=\
	asm.$O\
	cgocall.$O\
	chan.$O\
	closure.$O\
	float.$O\
	hashmap.$O\
	iface.$O\
	malloc.$O\
	mcache.$O\
	mcentral.$O\
	mem.$O\
	memmove.$O\
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
	runtime1.$O\
	rt0.$O\
	sema.$O\
	signal.$O\
	sigqueue.$O\
	slice.$O\
	string.$O\
	symtab.$O\
	sys.$O\
	thread.$O\
	traceback.$O\
	$(OFILES_$(GOARCH))\
	$(OFILES_$(GOOS))\

HFILES=\
	cgocall.h\
	runtime.h\
	hashmap.h\
	malloc.h\
	$(GOARCH)/asm.h\
	$(GOOS)/os.h\
	$(GOOS)/signals.h\
	$(GOOS)/$(GOARCH)/defs.h\

GOFILES+=$(GOFILES_$(GOOS))

include ../../Make.pkg

clean: clean-local

clean-local:
	rm -f cgo2c */asm.h runtime.acid.*

$(GOARCH)/asm.h: mkasmh.sh runtime.acid.$(GOARCH)
	./mkasmh.sh >$@.x
	mv -f $@.x $@

cgo2c: cgo2c.c
	$(QUOTED_GOBIN)/quietgcc -o $@ $<

%.c:	%.cgo cgo2c
	./cgo2c $< > $@.tmp
	mv -f $@.tmp $@

%.$O:	$(GOARCH)/%.c
	$(QUOTED_GOBIN)/$(CC) $(CFLAGS) $<

%.$O:	$(GOOS)/%.c
	$(QUOTED_GOBIN)/$(CC) $(CFLAGS) $<

%.$O:	$(GOOS)/$(GOARCH)/%.c
	$(QUOTED_GOBIN)/$(CC) $(CFLAGS) $<

%.$O:	$(GOARCH)/%.s $(GOARCH)/asm.h
	$(QUOTED_GOBIN)/$(AS) $<

%.$O:	$(GOOS)/$(GOARCH)/%.s $(GOARCH)/asm.h
	$(QUOTED_GOBIN)/$(AS) $<

# for discovering offsets inside structs when debugging
runtime.acid.$(GOARCH): runtime.h proc.c
	$(QUOTED_GOBIN)/$(CC) $(CFLAGS) -a proc.c >$@
