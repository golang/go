# Derived from Inferno libmach/mkfile
# http://code.google.com/p/inferno-os/source/browse/utils/libmach/mkfile
#
#	Copyright © 1994-1999 Lucent Technologies Inc.
#	Power PC support Copyright © 1995-2004 C H Forsyth (forsyth@terzarima.net).
#	Portions Copyright © 1997-1999 Vita Nuova Limited.
#	Portions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com).
#	Revisions Copyright © 2000-2004 Lucent Technologies Inc. and others.
#	Portions Copyright © 2009 The Go Authors.  All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

CFLAGS=-I$(GOROOT)/include
O=o

LIB=libmach_amd64.a
OFILES=\
	executable.$O\
	map.$O\
	obj.$O\
	swap.$O\
	sym.$O\
	access.$O\
	machdata.$O\
	setmach.$O\
	6.$O\
	8.$O\
	8db.$O\
	6obj.$O\
#	v.$O\
#	k.$O\
#	u.$O\
#	q.$O\
#	0.$O\
#	2.$O\
#	5.$O\
#	7.$O\
#	9.$O\
#	vdb.$O\
#	kdb.$O\
#	udb.$O\
#	qdb.$O\
#	2db.$O\
#	5db.$O\
#	7db.$O\
#	vobj.$O\
#	kobj.$O\
#	uobj.$O\
#	2obj.$O\
#	5obj.$O\
#	7obj.$O\
#	8obj.$O\
#	9obj.$O\
#	qobj.$O\
#	vcodas.$O\

HFILES=$(GOROOT)/include/mach_amd64.h elf.h macho.h obj.h

install: $(LIB)
	cp $(LIB) $(GOROOT)/lib

$(LIB): $(OFILES)
	ar rsc $(LIB) $(OFILES)

$(OFILES): $(HFILES)

clean:
	rm -f $(OFILES) $(LIB)
