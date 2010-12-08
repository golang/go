# Copyright 2010 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../../Make.inc

TARG=runtime/cgo

GOFILES=\
	cgo.go\

# Unwarranted chumminess with Make.pkg's cgo rules.
# Do not try this at home.
GCC_OFILES=\
	$(GOARCH).o\
	$(GOOS)_$(GOARCH).o\
	util.o\

OFILES=\
	iscgo.$O\
	_cgo_import.$O\
	$(GCC_OFILES)\

CGO_LDFLAGS=-lpthread

include ../../../Make.pkg

$(GOARCH).o: $(GOARCH).S
	$(HOST_CC) $(_CGO_CFLAGS_$(GOARCH)) -g -O2 -fPIC -o $@ -c $^

$(GOOS)_$(GOARCH).o: $(GOOS)_$(GOARCH).c
	$(HOST_CC) $(_CGO_CFLAGS_$(GOARCH)) -g -O2 -fPIC -o $@ -c $^

%.o: %.c
	$(HOST_CC) $(_CGO_CFLAGS_$(GOARCH)) -g -O2 -fPIC -o $@ -c $^
