# Copyright 2009 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# GNU Make syntax:
ifndef GOBIN
GOBIN=$(HOME)/bin
endif

all: $(TARG)

$(TARG): _go_.$O $(OFILES)
	$(LD) -o $@ _go_.$O $(OFILES)

_go_.$O: $(GOFILES)
	$(GC) -o $@ $(GOFILES)

install: $(GOBIN)/$(TARG)

$(GOBIN)/$(TARG): $(TARG)
	cp $(TARG) $@

clean:
	rm -f *.[$(OS)] $(TARG) $(CLEANFILES)

nuke:
	rm -f *.[$(OS)] $(TARG) $(CLEANFILES) $(GOBIN)/$(TARG)
