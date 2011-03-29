# Copyright 2010 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../Make.inc

TARG=install

clean:
	@true

install: install-gotry

install-%: %
	! test -f "$(GOBIN)"/$* || chmod u+w "$(GOBIN)"/$*
	sed 's`@@GOROOT@@`$(GOROOT_FINAL)`' $* >"$(GOBIN)"/$*
	chmod +x "$(GOBIN)"/$*

