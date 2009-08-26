# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include $(GOROOT)/src/Make.$(GOARCH)

TARG=unicode
GOFILES=\
	decimaldigit.go\
	digittables.go\
	letter.go\
	lettertables.go\

include $(GOROOT)/src/Make.pkg

tables:
	$(GC) maketables.go
	$(LD) -o maketables maketables.$O
	maketables --digits > digittables.go
	maketables > lettertables.go
	rm -f maketables
