# Copyright 2009 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

all: $(TARG)

# ugly hack to deal with whitespaces in $GOROOT
nullstring :=
space := $(nullstring) # a space at the end
QUOTED_GOROOT:=$(subst $(space),\ ,$(GOROOT))

include $(QUOTED_GOROOT)/src/Make.common

PREREQ+=$(patsubst %,%.make,$(DEPS))

$(TARG): _go_.$O $(OFILES)
	$(QUOTED_GOBIN)/$(LD) -o $@ _go_.$O $(OFILES)

_go_.$O: $(GOFILES) $(PREREQ)
	$(QUOTED_GOBIN)/$(GC) -o $@ $(GOFILES)

install: $(QUOTED_GOBIN)/$(TARG)

$(QUOTED_GOBIN)/$(TARG): $(TARG)
	cp -f $(TARG) $(QUOTED_GOBIN)

CLEANFILES+=$(TARG)

nuke: clean
	rm -f $(QUOTED_GOBIN)/$(TARG)
