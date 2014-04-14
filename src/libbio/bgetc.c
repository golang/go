/*
http://code.google.com/p/inferno-os/source/browse/libbio/bgetc.c

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

int
Bgetc(Biobuf *bp)
{
	int i;

loop:
	i = bp->icount;
	if(i != 0) {
		bp->icount = i+1;
		return bp->ebuf[i];
	}
	if(bp->state != Bractive) {
		if(bp->state == Bracteof)
			bp->state = Bractive;
		return Beof;
	}
	/*
	 * get next buffer, try to keep Bungetsize
	 * characters pre-catenated from the previous
	 * buffer to allow that many ungets.
	 */
	memmove(bp->bbuf-Bungetsize, bp->ebuf-Bungetsize, Bungetsize);
	i = (int)read(bp->fid, bp->bbuf, (size_t)bp->bsize);
	bp->gbuf = bp->bbuf;
	if(i <= 0) {
		bp->state = Bracteof;
		if(i < 0)
			bp->state = Binactive;
		return Beof;
	}
	if(i < bp->bsize) {
		memmove(bp->ebuf-i-Bungetsize, bp->bbuf-Bungetsize, (size_t)(i+Bungetsize));
		bp->gbuf = bp->ebuf-i;
	}
	bp->icount = -i;
	bp->offset += i;
	goto loop;
}

int
Bgetle2(Biobuf *bp)
{
	int l, h;

	l = Bgetc(bp);
	h = Bgetc(bp);
	return l|(h<<8);
}

int
Bgetle4(Biobuf *bp)
{
	int l, h;

	l = Bgetle2(bp);
	h = Bgetle2(bp);
	return (int)((uint32)l|((uint32)h<<16));
}

int
Bungetc(Biobuf *bp)
{

	if(bp->state == Bracteof)
		bp->state = Bractive;
	if(bp->state != Bractive)
		return Beof;
	bp->icount--;
	return 1;
}
