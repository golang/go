# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../../Make.inc

TARG=gob
GOFILES=\
	decode.go\
	decoder.go\
	doc.go\
	encode.go\
	encoder.go\
	error.go\
	type.go\

include ../../../Make.pkg

# Help for debugging. Requires adding debug.go to the gob package as well.

dump:	dump.$O
	$(LD) -o dump $<

dump.$O:	dump.go
	$(GC) $<
