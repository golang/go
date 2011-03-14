# Copyright 2009 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

ifeq ($(GOOS),windows)
TARG:=$(TARG).exe
endif

all: $(TARG)

include $(QUOTED_GOROOT)/src/Make.common

PREREQ+=$(patsubst %,%.make,$(DEPS))

$(TARG): _go_.$O
	$(LD) -o $@ _go_.$O

_go_.$O: $(GOFILES) $(PREREQ)
	$(GC) -o $@ $(GOFILES)

install: $(QUOTED_GOBIN)/$(TARG)

$(QUOTED_GOBIN)/$(TARG): $(TARG)
	cp -f $(TARG) $(QUOTED_GOBIN)

CLEANFILES+=$(TARG) _test _testmain.go

nuke: clean
	rm -f $(QUOTED_GOBIN)/$(TARG)

# for gotest
testpackage: _test/main.a

testpackage-clean:
	rm -f _test/main.a _gotest_.$O

_test/main.a: _gotest_.$O
	@mkdir -p _test
	rm -f $@
	gopack grc $@ _gotest_.$O

_gotest_.$O: $(GOFILES) $(GOTESTFILES)
	$(GC) -o $@ $(GOFILES) $(GOTESTFILES)

importpath:
	echo main
