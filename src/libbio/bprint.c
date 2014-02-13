/*
http://code.google.com/p/inferno-os/source/browse/libbio/bprint.c

	Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
	Revisions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com).  All rights reserved.
	Revisions Copyright © 2010 Google Inc.  All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include	<u.h>
#include	<libc.h>
#include	<bio.h>

int
Bprint(Biobuf *bp, char *fmt, ...)
{
	int n;
	va_list arg;

	va_start(arg, fmt);
	n = Bvprint(bp, fmt, arg);
	va_end(arg);
	return n;
}

static int
bflush(Fmt *f)
{
	Biobuf *bp;
	
	if(f->stop == nil)
		return 0;

	bp = f->farg;
	bp->ocount = (int)((char*)f->to - (char*)f->stop);
	if(Bflush(bp) < 0) {
		f->stop = nil;
		f->to = nil;
		return 0;
	}
	f->to = (char*)f->stop + bp->ocount;
	
	return 1;
}

int
Bvprint(Biobuf *bp, char *fmt, va_list arg)
{
	int n;
	Fmt f;
	
	memset(&f, 0, sizeof f);
	fmtlocaleinit(&f, nil, nil, nil);
	f.stop = bp->ebuf;
	f.to = (char*)f.stop + bp->ocount;
	f.flush = bflush;
	f.farg = bp;

	n = fmtvprint(&f, fmt, arg);

	if(f.stop != nil)
		bp->ocount = (int)((char*)f.to - (char*)f.stop);

	return n;
}
