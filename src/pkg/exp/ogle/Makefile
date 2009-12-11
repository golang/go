# Copyright 2009 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../../Make.$(GOARCH)

TARG=exp/ogle
GOFILES=\
	abort.go\
	arch.go\
	cmd.go\
	event.go\
	frame.go\
	goroutine.go\
	rruntime.go\
	rtype.go\
	rvalue.go\
	process.go\
	vars.go\

CLEANFILES+=ogle

include ../../../Make.pkg

main.$O: main.go package
	$(QUOTED_GOBIN)/$(GC) -I_obj $<

ogle: main.$O
	$(QUOTED_GOBIN)/$(LD) -L_obj -o $@ $<
