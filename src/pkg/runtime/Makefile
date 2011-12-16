# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../Make.inc

TARG=runtime

GOFILES=\
	debug.go\
	error.go\
	extern.go\
	mem.go\
	sig.go\
	softfloat64.go\
	type.go\
	zgoarch_$(GOARCH).go\
	zgoos_$(GOOS).go\
	zruntime_defs_$(GOOS)_$(GOARCH).go\
	zversion.go\

OFILES_darwin=\
	lock_sema.$O\

OFILES_freebsd=\
	lock_futex.$O\

OFILES_linux=\
	lock_futex.$O\

OFILES_netbsd=\
	lock_sema.$O\

OFILES_openbsd=\
	lock_sema.$O\

OFILES_plan9=\
	lock_sema.$O\

OFILES_windows=\
	callback_windows_$(GOARCH).$O\
	lock_sema.$O\
	zsyscall_windows_$(GOARCH).$O\

# 386-specific object files
OFILES_386=\
	vlop_386.$O\
	vlrt_386.$O\
	traceback_x86.$O\

# amd64-specific object files
OFILES_amd64=\
	traceback_x86.$O\

# arm-specific object files
OFILES_arm=\
	memset_arm.$O\
	softfloat_arm.$O\
	vlop_arm.$O\
	vlrt_arm.$O\
	traceback_x86.$O\

OFILES=\
	$(OFILES_$(GOARCH))\
	$(OFILES_$(GOOS))\
	alg.$O\
	asm_$(GOARCH).$O\
	atomic_$(GOARCH).$O\
	cgocall.$O\
	chan.$O\
	closure_$(GOARCH).$O\
	complex.$O\
	cpuprof.$O\
	float.$O\
	hashmap.$O\
	iface.$O\
	mcache.$O\
	mcentral.$O\
	mem_$(GOOS).$O\
	memmove_$(GOARCH).$O\
	mfinal.$O\
	mfixalloc.$O\
	mgc0.$O\
	mheap.$O\
	msize.$O\
	print.$O\
	proc.$O\
	rt0_$(GOOS)_$(GOARCH).$O\
	rune.$O\
	runtime.$O\
	signal_$(GOOS)_$(GOARCH).$O\
	slice.$O\
	symtab.$O\
	sys_$(GOOS)_$(GOARCH).$O\
	thread_$(GOOS).$O\
	zmalloc_$(GOARCH).$O\
	zmprof_$(GOARCH).$O\
	zruntime1_$(GOARCH).$O\
	zsema_$(GOARCH).$O\
	zsigqueue_$(GOARCH).$O\
	zstring_$(GOARCH).$O\
	ztime_$(GOARCH).$O\

AUTOHFILES=\
	arch_GOARCH.h\
	os_GOOS.h\
	signals_GOOS.h\
	defs_GOOS_GOARCH.h\
	zasm_GOOS_GOARCH.h\

HFILES=\
	cgocall.h\
	runtime.h\
	hashmap.h\
	malloc.h\
	stack.h\
	asm_$(GOARCH).h\
	$(AUTOHFILES)\

GOFILES+=$(GOFILES_$(GOOS))

# For use by cgo.
INSTALLFILES=$(pkgdir)/runtime.h $(pkgdir)/cgocall.h

# Go tool will do this for package runtime.
AFLAGS+=-DGOOS_$(GOOS) -DGOARCH_$(GOARCH)
CFLAGS+=-DGOOS_$(GOOS) -DGOARCH_$(GOARCH)

# special, out of the way compiler flag that means "add runtime metadata to output"
GC+= -+

include ../../Make.pkg

$(pkgdir)/%.h: %.h
	@test -d $(QUOTED_GOROOT)/pkg && mkdir -p $(pkgdir)
	cp $< "$@"

clean: clean-local

clean-local:
	rm -f $(AUTOHFILES)
	make -f Makefile.auto clean

arch_GOARCH.h: arch_$(GOARCH).h
	cp $^ $@

defs_GOOS_GOARCH.h: defs_$(GOOS)_$(GOARCH).h
	cp $^ $@

os_GOOS.h: os_$(GOOS).h
	cp $^ $@

signals_GOOS.h: signals_$(GOOS).h
	cp $^ $@

zasm_GOOS_GOARCH.h: zasm_$(GOOS)_$(GOARCH).h
	cp $^ $@
