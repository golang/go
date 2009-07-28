# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../Make.conf

TARG=godefs
OFILES=\
	main.$O\
	stabs.$O\
	util.$O\

HFILES=a.h

$(TARG): $(OFILES)
	$(LD) -o $(TARG) -L$(GOROOT)/lib $(OFILES) -lbio -l9

clean:
	rm -f $(OFILES) $(TARG)

install: $(TARG)
	cp $(TARG) $(GOBIN)/$(TARG)

$(OFILES): $(HFILES)
