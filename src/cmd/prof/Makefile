# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../Make.conf

# The directory is db because the source is portable and general.
# We call the binary 6prof to avoid confusion and because this binary
# is linked only with amd64 and x86 support.

TARG=6prof
OFILES=\
	main.$O\

#HFILES=\
#	defs.h\
#	fns.h\

$(TARG): $(OFILES)
	$(LD) -o $(TARG) -L$(GOROOT)/lib $(OFILES) -lmach_amd64 -lbio -l9

clean:
	rm -f $(OFILES) $(TARG)

install: $(TARG)
	cp $(TARG) $(BIN)/$(TARG)

$(OFILES): $(HFILES)
