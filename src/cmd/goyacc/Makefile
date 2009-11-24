# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../Make.$(GOARCH)

TARG=goyacc
GOFILES=\
	goyacc.go\

include ../../Make.cmd

units: goyacc units.y
	./goyacc units.y
	$(GC) y.go
	$(LD) -o units y.$O

