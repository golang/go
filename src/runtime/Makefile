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
	rt1_$(GOARCH)_$(GOOS).$O\
	rt2_$(GOARCH).$O\

OFILES=$(RT0OFILES) $(LIBOFILES)
HFILES=

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

%.$O:	%.s
	$(AS) $<
