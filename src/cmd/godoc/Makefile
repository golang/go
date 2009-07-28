# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include $(GOROOT)/src/Make.conf
include $(GOROOT)/src/Make.$(GOARCH)

TARG=godoc
OFILES=\
	godoc.$O\

$(TARG): $(OFILES)
	$(LD) -o $(TARG) $(OFILES)

clean:
	rm -f $(OFILES) $(TARG)

install: $(TARG)
	cp $(TARG) $(GOBIN)/$(TARG)

%.$O:	%.go
	$(GC) $<
