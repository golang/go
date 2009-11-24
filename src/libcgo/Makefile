# Copyright 2009 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# ugly hack to deal with whitespaces in $GOROOT
nullstring :=
space := $(nullstring) # a space at the end
QUOTED_GOROOT=$(subst $(space),\ ,$(GOROOT))

all: libcgo.so

install: $(QUOTED_GOROOT)/pkg/$(GOOS)_$(GOARCH)/libcgo.so

OFILES=\
	$(GOOS)_$(GOARCH).o\
	$(GOARCH).o\
	util.o\

CFLAGS_386=-m32
CFLAGS_amd64=-m64

LDFLAGS_linux=-shared -lpthread -lm
LDFLAGS_darwin=-dynamiclib -Wl,-undefined,dynamic_lookup /usr/lib/libpthread.dylib
LDFLAGS_freebsd=-pthread -shared -lm

%.o: %.c
	gcc $(CFLAGS_$(GOARCH)) -O2 -fPIC -o $@ -c $*.c

%.o: %.S
	gcc $(CFLAGS_$(GOARCH)) -O2 -fPIC -o $@ -c $*.S

libcgo.so: $(OFILES)
	gcc $(CFLAGS_$(GOARCH)) -o libcgo.so $(OFILES) $(LDFLAGS_$(GOOS))

$(QUOTED_GOROOT)/pkg/$(GOOS)_$(GOARCH)/libcgo.so: libcgo.so
	cp libcgo.so $(QUOTED_GOROOT)/pkg/$(GOOS)_$(GOARCH)

clean:
	rm -f *.o *.so

