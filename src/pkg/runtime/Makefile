# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../Make.inc

AUTOHFILES=\
	arch_GOARCH.h\
	os_GOOS.h\
	signals_GOOS.h\
	defs_GOOS_GOARCH.h\
	zasm_GOOS_GOARCH.h\

all:
	@echo use make install, make clean

pkgdir=$(QUOTED_GOROOT)/pkg/$(GOOS)_$(GOARCH)

$(pkgdir)/%.h: %.h
	@test -d $(QUOTED_GOROOT)/pkg && mkdir -p $(pkgdir)
	cp $< "$@"

install: $(pkgdir)/runtime.h $(pkgdir)/cgocall.h

clean: clean-local

clean-local:
	rm -f $(AUTOHFILES) runtime_defs.go version*.go asm_*.h

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
