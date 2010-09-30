# Copyright 2009 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

ifeq ($(GOOS),nacl)

# Nothing for NaCl
all clean install:
	@true

else ifeq ($(GOARCH),arm)

# Nothing for ARM - usually cross compiling anyway
all clean install:
	@true

else
include ../Make.inc

all: libcgo.so

install: $(QUOTED_GOROOT)/pkg/$(GOOS)_$(GOARCH)/libcgo.so

OFILES=\
	$(GOOS)_$(GOARCH).o\
	$(GOARCH).o\
	util.o\

HOST_CFLAGS_386=-m32
HOST_CFLAGS_amd64=-m64

LDFLAGS_darwin=-dynamiclib -Wl,-undefined,dynamic_lookup /usr/lib/libpthread.dylib
LDFLAGS_freebsd=-shared -lm -pthread
LDFLAGS_linux=-shared -lm -lpthread
LDFLAGS_nacl=-shared -lm -lpthread
LDFLAGS_windows=-shared -lm -mthreads

%.o: %.c
	$(HOST_CC) $(HOST_CFLAGS_$(GOARCH)) -g -O2 -fPIC -o $@ -c $*.c

%.o: %.S
	$(HOST_CC) $(HOST_CFLAGS_$(GOARCH)) -g -O2 -fPIC -o $@ -c $*.S

libcgo.so: $(OFILES)
	$(HOST_CC) $(HOST_CFLAGS_$(GOARCH)) -o libcgo.so $(OFILES) $(LDFLAGS_$(GOOS))

$(QUOTED_GOROOT)/pkg/$(GOOS)_$(GOARCH)/libcgo.so: libcgo.so
	cp libcgo.so $(QUOTED_GOROOT)/pkg/$(GOOS)_$(GOARCH)

clean:
	rm -f *.o *.so

endif
