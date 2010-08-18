# Copyright 2010 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../../src/Make.inc

all: index.html

# ugly hack to deal with whitespaces in $GOROOT
nullstring :=
space := $(nullstring) # a space at the end
QUOTED_GOROOT:=$(subst $(space),\ ,$(GOROOT))

include $(QUOTED_GOROOT)/src/Make.common

CLEANFILES+=index.html srcextract.bin htmlify.bin

index.html: srcextract.bin htmlify.bin
	awk '/^!/{system(substr($$0,2)); next} {print}' "$$@" < wiki.html > index.html

test: final.bin
	./test.sh
	rm -f final.6 final.bin

%.bin: %.$O
	$(QUOTED_GOBIN)/$(LD) -o $@ $<
%.$O: 
	$(QUOTED_GOBIN)/$(GC) $*.go

