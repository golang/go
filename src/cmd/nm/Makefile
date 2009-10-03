# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../Make.conf

# The directory is nm because the source is portable and general.
# We call the binary 6nm to avoid confusion and because this binary
# is linked only with amd64 and x86 support.

TARG=6nm
OFILES=\
	nm.$O\

$(TARG): $(OFILES)
	$(LD) -o $(TARG) -L$(GOROOT)/lib $(OFILES) -lmach -lbio -l9

clean:
	rm -f *.$O $(TARG)

install: $(TARG)
	cp $(TARG) $(GOBIN)/$(TARG)

$(OFILES): $(HFILES)
