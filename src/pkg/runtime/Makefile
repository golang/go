# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../Make.inc

TARG=runtime

# Set SIZE to 32 or 64.
SIZE_386=32
SIZE_amd64=64
SIZE_arm=32
SIZE=$(SIZE_$(GOARCH))

CFLAGS_windows=-D__WINDOWS__
CFLAGS=-I$(GOOS) -I$(GOARCH) -I$(GOOS)/$(GOARCH) -FVw $(CFLAGS_$(GOARCH)) $(CFLAGS_$(GOOS))

GOFILES=\
	debug.go\
	error.go\
	extern.go\
	mem.go\
	sig.go\
	softfloat64.go\
	type.go\
	version.go\
	version_$(GOOS).go\
	version_$(GOARCH).go\
	runtime_defs.go\

CLEANFILES+=version.go version_*.go

OFILES_darwin=\
	lock_sema.$O\

OFILES_freebsd=\
	lock_futex.$O\

OFILES_linux=\
	lock_futex.$O\

OFILES_openbsd=\
	lock_sema.$O\

OFILES_plan9=\
	lock_sema.$O\

OFILES_windows=\
	callback.$O\
	lock_sema.$O\
	syscall.$O\

# 386-specific object files
OFILES_386=\
	vlop.$O\
	vlrt.$O\

# arm-specific object files
OFILES_arm=\
	memset.$O\
	softfloat.$O\
	vlop.$O\
	vlrt.$O\

OFILES=\
	asm.$O\
	atomic.$O\
	cgocall.$O\
	chan.$O\
	closure.$O\
	cpuprof.$O\
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
	mprof.$O\
	msize.$O\
	print.$O\
	proc.$O\
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
	time.$O\
	traceback.$O\
	$(OFILES_$(GOARCH))\
	$(OFILES_$(GOOS))\

HFILES=\
	cgocall.h\
	runtime.h\
	hashmap.h\
	malloc.h\
	stack.h\
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
	cp $< "$@"

clean: clean-local

clean-local:
	rm -f goc2c mkversion version.go */asm.h runtime.acid.* runtime_defs.go $$(ls *.goc | sed 's/goc$$/c/')

$(GOARCH)/asm.h: mkasmh.sh runtime.acid.$(GOARCH)
	./mkasmh.sh >$@.x
	mv -f $@.x $@

goc2c: goc2c.c
	quietgcc -o $@ -I "$(GOROOT)/include" $< "$(GOROOT)/lib/lib9.a"

mkversion: mkversion.c
	quietgcc -o $@ -I "$(GOROOT)/include" $< "$(GOROOT)/lib/lib9.a"

version.go: mkversion
	GOROOT="$(GOROOT_FINAL)" ./mkversion >version.go

version_$(GOARCH).go:
	(echo 'package runtime'; echo 'const theGoarch = "$(GOARCH)"') >$@

version_$(GOOS).go:
	(echo 'package runtime'; echo 'const theGoos = "$(GOOS)"') >$@

%.c:	%.goc goc2c
	./goc2c "`pwd`/$<" > $@.tmp
	mv -f $@.tmp $@

%.$O:	$(GOARCH)/%.c $(HFILES)
	$(CC) $(CFLAGS) $<

%.$O:	$(GOOS)/%.c $(HFILES)
	$(CC) $(CFLAGS) $<

%.$O:	$(GOOS)/$(GOARCH)/%.c $(HFILES)
	$(CC) $(CFLAGS) $<

%.$O:	$(GOARCH)/%.s $(GOARCH)/asm.h
	$(AS) $<

%.$O:	$(GOOS)/$(GOARCH)/%.s $(GOARCH)/asm.h
	$(AS) $<

# for discovering offsets inside structs when debugging
runtime.acid.$(GOARCH): runtime.h proc.c
	$(CC) $(CFLAGS) -a proc.c >$@

# 386 traceback is really amd64 traceback
ifeq ($(GOARCH),386)
traceback.$O:	amd64/traceback.c
	$(CC) $(CFLAGS) $<
endif

runtime_defs.go: proc.c iface.c hashmap.c chan.c
	CC="$(CC)" CFLAGS="$(CFLAGS)" ./mkgodefs.sh $^ > $@.x
	mv -f $@.x $@
