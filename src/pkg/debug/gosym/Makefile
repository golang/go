# Copyright 2009 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../../Make.inc

TARG=debug/gosym
GOFILES=\
	pclntab.go\
	symtab.go\

include ../../../Make.pkg

test: make-pclinetest

testshort: make-pclinetest

make-pclinetest:
	@if [ "`uname`-`uname -m`" = Linux-x86_64 -a $(GOARCH) = amd64 ]; then mkdir -p _test && $(AS) pclinetest.s && $(LD) -E main -o _test/pclinetest pclinetest.$O; fi
