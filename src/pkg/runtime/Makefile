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
CFLAGS_windows=-D__WINDOWS__
CFLAGS=-I$(GOOS) -I$(GOARCH) -I$(GOOS)/$(GOARCH) -wF $(CFLAGS_$(SIZE)) $(CFLAGS_$(GOARCH)) $(CFLAGS_$(GOOS))

GOFILES=\
	error.go\
	extern.go\
	type.go\
	version.go\

GOFILES_tiny=\
	tiny/io.go\

OFILES_windows=\
	syscall.$O\

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
	complex.$O\
	hashmap.$O\
	iface.$O\
	malloc.$O\
	mcache.$O\
	mcentral.$O\
	mem.$O\
	memmove.$O\
	mfinal.$O\
	mfixalloc.$O\
	mgc0.$O\
	mheap.$O\
	mheapmap$(SIZE).$O\
	mprof.$O\
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

# For use by cgo.
INSTALLFILES=$(pkgdir)/runtime.h $(pkgdir)/cgocall.h

# special, out of the way compiler flag that means "add runtime metadata to output"
GC+= -+

include ../../Make.pkg

$(pkgdir)/%.h: %.h
	@test -d $(QUOTED_GOROOT)/pkg && mkdir -p $(pkgdir)
	cp $< $@

clean: clean-local

clean-local:
	rm -f goc2c mkversion version.go */asm.h runtime.acid.*

$(GOARCH)/asm.h: mkasmh.sh runtime.acid.$(GOARCH)
	./mkasmh.sh >$@.x
	mv -f $@.x $@

goc2c: goc2c.c
	$(QUOTED_GOBIN)/quietgcc -o $@ $<

mkversion: mkversion.c
	$(QUOTED_GOBIN)/quietgcc -o $@ -I "$(GOROOT)/include" $< "$(GOROOT)/lib/lib9.a"

version.go: mkversion
	./mkversion >version.go

%.c:	%.goc goc2c
	./goc2c $< > $@.tmp
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

# 386 traceback is really amd64 traceback
ifeq ($(GOARCH),386)
traceback.$O:	amd64/traceback.c
	$(QUOTED_GOBIN)/$(CC) $(CFLAGS) $<
endif

# NaCl closure is special.
ifeq ($(GOOS),nacl)
closure.$O: nacl/$(GOARCH)/closure.c
	$(QUOTED_GOBIN)/$(CC) $(CFLAGS) $<
endif
