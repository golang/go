/*
http://code.google.com/p/inferno-os/source/browse/libbio/bwrite.c

	Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
	Revisions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com).  All rights reserved.

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

long
Bwrite(Biobuf *bp, void *ap, long count)
{
	long c;
	unsigned char *p;
	int i, n, oc;

	p = ap;
	c = count;
	oc = bp->ocount;

	while(c > 0) {
		n = -oc;
		if(n > c)
			n = (int)c;
		if(n == 0) {
			if(bp->state != Bwactive)
				return Beof;
			i = (int)write(bp->fid, bp->bbuf, (size_t)bp->bsize);
			if(i != bp->bsize) {
				bp->state = Binactive;
				return Beof;
			}
			bp->offset += i;
			oc = -bp->bsize;
			continue;
		}
		memmove(bp->ebuf+oc, p, (size_t)n);
		oc += n;
		c -= n;
		p += n;
	}
	bp->ocount = oc;
	return count-c;
}
