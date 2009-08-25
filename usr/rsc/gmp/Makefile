# Copyright 2009 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# FFI demo

include ../../../src/Make.$(GOARCH)

all: gmp.a gmp.so

gcc.o: gcc.c
	gcc -fPIC -O2 -o gcc.o -c gcc.c

gmp.so: gcc.o
	gcc -shared -o gmp.so gcc.o -L$(GOROOT)/pkg/$(GOOS)_$(GOARCH) -lcgo -lgmp

gmp.a: 6c.6 go.6
	gopack grc gmp.a 6c.6 go.6

# from pkg/runtime/Makefile: TODO(rsc): how to deal with this?
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

6c.6: 6c.c
	6c -FVw $(CFLAGS) -I$(GOROOT)/src/pkg/runtime 6c.c

go.6: go.go
	6g go.go

PKG=$(GOROOT)/pkg/$(GOOS)_$(GOARCH)

install: $(PKG)/gmp.so $(PKG)/gmp.a

$(PKG)/gmp.so: gmp.so
	cp gmp.so $@

$(PKG)/gmp.a: gmp.a
	cp gmp.a $@

clean:
	rm -f *.6 *.o *.so *.a

