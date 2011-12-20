# Copyright 2010 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../../Make.inc

TARG=runtime/cgo

GOFILES=\
	cgo.go\

ifeq ($(CGO_ENABLED),1)

# Unwarranted chumminess with Make.pkg's cgo rules.
# Do not try this at home.
CGO_OFILES=\
	gcc_$(GOARCH).o\
	gcc_$(GOOS)_$(GOARCH).o\
	gcc_util.o\

ifeq ($(GOOS),windows)
CGO_LDFLAGS=-lm -mthreads
else
CGO_LDFLAGS=-lpthread
CGO_OFILES+=gcc_setenv.o\

endif

OFILES=\
	iscgo.$O\
	callbacks.$O\
	_cgo_import.$O\
	$(CGO_OFILES)\

ifeq ($(GOOS),freebsd)
OFILES+=\
	freebsd.$O\

endif

endif

include ../../../Make.pkg

ifeq ($(CGO_ENABLED),1)
_cgo_defun.c:
	echo >$@

_cgo_main.c:
	echo 'int main() { return 0; }' >$@
endif

gcc_$(GOARCH).o: gcc_$(GOARCH).S
	$(HOST_CC) $(_CGO_CFLAGS_$(GOARCH)) -g -O2 -fPIC -o $@ -c $^

gcc_$(GOOS)_$(GOARCH).o: gcc_$(GOOS)_$(GOARCH).c
	$(HOST_CC) $(_CGO_CFLAGS_$(GOARCH)) -g -O2 -fPIC -o $@ -c $^

gcc_%.o: gcc_%.c
	$(HOST_CC) $(_CGO_CFLAGS_$(GOARCH)) -g -O2 -fPIC -o $@ -c $^
