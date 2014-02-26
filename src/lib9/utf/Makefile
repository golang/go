# Copyright 2010 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# The library is built by the Makefile in the parent directory.
# This Makefile only builds mkrunetype.
# GOROOT, GOOS, and GOARCH must be set explicitly.

TARG=mkrunetype

UnicodeData-%.txt:
	curl http://www.unicode.org/Public/$*/ucd/UnicodeData.txt >_$@
	mv _$@ $@

mkrunetype: mkrunetype.c
	cc -I../../../include -o mkrunetype -L$(GOROOT)/pkg/obj/$(GOOS)_$(GOARCH)/ mkrunetype.c -l9

runetypebody-%.h: mkrunetype UnicodeData-%.txt
	mkrunetype -p UnicodeData-$*.txt >_$@
	mv _$@ $@

CLEANFILES+=UnicodeData.txt

UNICODE_VERSION=6.3.0

test: mkrunetype UnicodeData-$(UNICODE_VERSION).txt
	mkrunetype -c UnicodeData-$(UNICODE_VERSION).txt

clean:
	rm -f UnicodeData.txt mkrunetype
