# Copyright 2009 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

ifeq ($(GOOS),windows)
TARG:=$(TARG).exe
endif

ifeq ($(TARGDIR),)
TARGDIR:=$(QUOTED_GOBIN)
endif

all: $(TARG)

include $(QUOTED_GOROOT)/src/Make.common

PREREQ+=$(patsubst %,%.make,$(DEPS))

$(TARG): _go_.$O
	$(LD) $(LDIMPORTS) -o $@ _go_.$O

_go_.$O: $(GOFILES) $(PREREQ)
	$(GC) $(GCFLAGS) $(GCIMPORTS) -o $@ $(GOFILES)

install: $(TARGDIR)/$(TARG)

$(TARGDIR)/$(TARG): $(TARG)
	mkdir -p $(TARGDIR) && cp -f $(TARG) $(TARGDIR)

CLEANFILES+=$(TARG) _test _testmain.go test.out build.out

nuke: clean
	rm -f $(TARGDIR)/$(TARG)

# for gotest
testpackage: _test/main.a

testpackage-clean:
	rm -f _test/main.a _gotest_.$O

_test/main.a: _gotest_.$O
	@mkdir -p _test
	rm -f $@
	gopack grc $@ _gotest_.$O

_gotest_.$O: $(GOFILES) $(GOTESTFILES)
	$(GC) $(GCFLAGS) $(GCIMPORTS) -o $@ $(GOFILES) $(GOTESTFILES)

importpath:
	echo main
