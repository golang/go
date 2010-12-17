# Copyright 2010 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../../Make.inc

ENABLED:=1

ifeq ($(GOARCH),arm)
ENABLED:=0
endif

TARG=runtime/cgo

GOFILES=\
	cgo.go\

ifeq ($(ENABLED),1)

# Unwarranted chumminess with Make.pkg's cgo rules.
# Do not try this at home.
CGO_OFILES=\
	$(GOARCH).o\
	$(GOOS)_$(GOARCH).o\
	util.o\

OFILES=\
	iscgo.$O\
	_cgo_import.$O\
	$(CGO_OFILES)\

CGO_LDFLAGS=-lpthread

ifeq ($(GOOS),freebsd)
OFILES+=\
	freebsd.$O\

endif

endif

include ../../../Make.pkg

ifeq ($(ENABLED),1)
_cgo_defun.c:
	echo >$@

_cgo_main.c:
	echo 'int main() { return 0; }' >$@
	echo 'void *crosscall2;' >>$@
endif

$(GOARCH).o: $(GOARCH).S
	$(HOST_CC) $(_CGO_CFLAGS_$(GOARCH)) -g -O2 -fPIC -o $@ -c $^

$(GOOS)_$(GOARCH).o: $(GOOS)_$(GOARCH).c
	$(HOST_CC) $(_CGO_CFLAGS_$(GOARCH)) -g -O2 -fPIC -o $@ -c $^

%.o: %.c
	$(HOST_CC) $(_CGO_CFLAGS_$(GOARCH)) -g -O2 -fPIC -o $@ -c $^
