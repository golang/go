# Copyright 2010 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# The library is built by the Makefile in the parent directory.
# This Makefile only builds mkrunetype.

include ../../Make.inc
O:=$(HOST_O)

TARG=mkrunetype

OFILES=\
	mkrunetype.$O\

include ../../Make.ccmd

UnicodeData-%.txt:
	curl http://www.unicode.org/Public/$*/ucd/UnicodeData.txt >_$@
	mv _$@ $@

runetypebody-%.c: mkrunetype UnicodeData-%.txt
	mkrunetype -p UnicodeData-$*.txt >_$@
	mv _$@ $@

CLEANFILES+=UnicodeData.txt

UNICODE_VERSION=6.0.0

test: mkrunetype UnicodeData-$(UNICODE_VERSION).txt
	mkrunetype -c UnicodeData-$(UNICODE_VERSION).txt

