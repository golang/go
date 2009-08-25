# Copyright 2009 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# FFI demo

all: fib.a fib.so

gcc.o: gcc.c
	gcc -fPIC -O2 -o gcc.o -c gcc.c

fib.so: gcc.o
	gcc -shared -o fib.so gcc.o -L$(GOROOT)/pkg/$(GOOS)_$(GOARCH) -lcgo

fib.a: 6c.6 go.6
	gopack grc fib.a 6c.6 go.6

6c.6: 6c.c
	6c -FVw -I$(GOROOT)/src/pkg/runtime 6c.c

go.6: go.go
	6g go.go

PKG=$(GOROOT)/pkg/$(GOOS)_$(GOARCH)

install: $(PKG)/fib.so $(PKG)/fib.a

$(PKG)/fib.so: fib.so
	cp fib.so $@

$(PKG)/fib.a: fib.a
	cp fib.a $@

clean:
	rm -f *.6 *.o *.so *.a

