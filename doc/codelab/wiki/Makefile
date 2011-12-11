# Copyright 2010 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../../src/Make.inc

all: index.html

include ../../../src/Make.common

CLEANFILES+=srcextract.bin htmlify.bin get.bin

index.html: wiki.html srcextract.bin htmlify.bin
	PATH=.:$$PATH awk '/^!/{system(substr($$0,2)); next} {print}' < wiki.html | tr -d '\r' > index.html

test: get.bin
	bash ./test.sh
	rm -f get.6 get.bin

%.bin: %.$O
	$(LD) -o $@ $<

%.$O: %.go
	$(GC) $*.go

