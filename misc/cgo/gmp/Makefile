# Copyright 2009 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include $(GOROOT)/src/Make.$(GOARCH)

TARG=gmp
CGOFILES=\
	gmp.go

CGO_LDFLAGS=-lgmp

# Can have plain GOFILES too, but this example doesn't.

include $(GOROOT)/src/Make.pkg

# Simple test program

pidigits.$O: install pidigits.go
	$(GC) pidigits.go

pidigits: pidigits.$O
	$(LD) -o $@ pidigits.$O
