# Copyright 2009 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# not linked into build for now

TARG=libcgo.so

all: libcgo.so

cgocall.o: cgocall.c
	gcc -O2 -fPIC -o cgocall.o -c cgocall.c

libcgo.so: cgocall.o
	gcc -shared -o libcgo.so cgocall.o -lpthread -lm

install: $(GOROOT)/pkg/$(GOOS)_$(GOARCH)/libcgo.so

$(GOROOT)/pkg/$(GOOS)_$(GOARCH)/libcgo.so: libcgo.so
	cp libcgo.so $@

clean:
	rm -f *.o *.so

