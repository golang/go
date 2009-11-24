# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../Make.$(GOARCH)

TARG=ebnflint
GOFILES=\
	ebnflint.go\

include ../../Make.cmd

test: $(TARG)
	$(TARG) -start="SourceFile" "$(GOROOT)"/doc/go_spec.html

