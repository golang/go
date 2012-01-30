# Copyright 2009 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

ifeq ($(GOOS),windows)
TARG:=$(TARG).exe
endif

# Tools always go into $GOROOT/bin/go-tool
TOOLDIR=$(QUOTED_GOROOT)/bin/go-tool

all: $(TARG)

include $(QUOTED_GOROOT)/src/Make.common

PREREQ+=$(patsubst %,%.make,$(DEPS))

$(TARG): _go_.$O
	$(LD) $(LDIMPORTS) -o $@ _go_.$O

_go_.$O: $(GOFILES) $(PREREQ)
	$(GC) $(GCFLAGS) $(GCIMPORTS) -o $@ $(GOFILES)

install: $(TOOLDIR)/$(TARG)

$(TOOLDIR)/$(TARG): $(TARG)
	mkdir -p $(TOOLDIR) && cp -f $(TARG) $(TOOLDIR)

CLEANFILES+=$(TARG) _test _testmain.go test.out build.out

nuke: clean
	rm -f $(TOOLDIR)/$(TARG)
