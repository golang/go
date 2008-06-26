# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

CFLAGS=-I$(GOROOT)/include
BIN=$(HOME)/bin
O=6

# The directory is ar because the source is portable and general.
# We call the binary 6ar to avoid confusion and because this binary
# is linked only with amd64 and x86 support.

TARG=$(O)ar
OFILES=\
	ar.$O\

$(TARG): $(OFILES)
	cc -o $(TARG) -L$(GOROOT)/lib $(OFILES) -lmach_amd64 -lbio -l9

clean:
	rm -f $(OFILES) $(TARG)

install: $(TARG)
	cp $(TARG) $(BIN)/$(TARG)
