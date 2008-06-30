# Copyright 2009 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

CFLAGS=
O=6
CC=$(O)c
AS=$(O)a

LIB=lib_$(GOARCH)_$(GOOS).a

RT0OFILES=\
	rt0_$(GOARCH)_$(GOOS).$O\

LIBOFILES=\
	rt0_$(GOARCH).$O\
	rt1_$(GOARCH)_$(GOOS).$O\
	rt2_$(GOARCH).$O\
	sys_$(GOARCH)_$(GOOS).$O\
	runtime.$O\
	runtime_map.$O\
	runtime_print.$O\
	runtime_string.$O\
	sys_file.$O\

OFILES=$(RT0OFILES) $(LIBOFILES)
OS_H=$(GOARCH)_$(GOOS).h
HFILES=runtime.h $(OS_H_)

install: rt0 $(LIB)
	cp $(RT0OFILES) $(GOROOT)/lib
	cp $(LIB) $(GOROOT)/lib

rt0:	$(RT0OFILES)

$(LIB): $(LIBOFILES)
	$(O)ar rc $(LIB) $(LIBOFILES)

$(OFILES): $(HFILES)

clean:
	rm -f *.$(O) *.a

%.$O:	%.c
	$(CC) $<

sys_file.$O:	sys_file.c sys_types.h $(OS_H)
	$(CC) -D$(GOARCH)_$(GOOS) $<

%.$O:	%.s
	$(AS) $<
