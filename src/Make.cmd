# Copyright 2009 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# GNU Make syntax:
ifndef GOBIN
GOBIN=$(HOME)/bin
endif

# ugly hack to deal with whitespaces in $GOBIN
nullstring :=
space := $(nullstring) # a space at the end
QUOTED_GOBIN=$(subst $(space),\ ,$(GOBIN))

all: $(TARG)

$(TARG): _go_.$O $(OFILES)
	$(QUOTED_GOBIN)/$(LD) -o $@ _go_.$O $(OFILES)

_go_.$O: $(GOFILES)
	$(QUOTED_GOBIN)/$(GC) -o $@ $(GOFILES)

install: $(QUOTED_GOBIN)/$(TARG)

$(QUOTED_GOBIN)/$(TARG): $(TARG)
	cp -f $(TARG) $(QUOTED_GOBIN)

clean:
	rm -f *.[$(OS)] $(TARG) $(CLEANFILES)

nuke:
	rm -f *.[$(OS)] $(TARG) $(CLEANFILES) $(QUOTED_GOBIN)/$(TARG)
