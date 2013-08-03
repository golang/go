/*
http://code.google.com/p/inferno-os/source/browse/libbio/brdline.c

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

void*
Brdline(Biobuf *bp, int delim)
{
	char *ip, *ep;
	int i, j;

	i = -bp->icount;
	if(i == 0) {
		/*
		 * eof or other error
		 */
		if(bp->state != Bractive) {
			if(bp->state == Bracteof)
				bp->state = Bractive;
			bp->rdline = 0;
			bp->gbuf = bp->ebuf;
			return 0;
		}
	}

	/*
	 * first try in remainder of buffer (gbuf doesn't change)
	 */
	ip = (char*)bp->ebuf - i;
	ep = memchr(ip, delim, (size_t)i);
	if(ep) {
		j = (int)((ep - ip) + 1);
		bp->rdline = j;
		bp->icount += j;
		return ip;
	}

	/*
	 * copy data to beginning of buffer
	 */
	if(i < bp->bsize)
		memmove(bp->bbuf, ip, (size_t)i);
	bp->gbuf = bp->bbuf;

	/*
	 * append to buffer looking for the delim
	 */
	ip = (char*)bp->bbuf + i;
	while(i < bp->bsize) {
		j = (int)read(bp->fid, ip, (size_t)(bp->bsize-i));
		if(j <= 0) {
			/*
			 * end of file with no delim
			 */
			memmove(bp->ebuf-i, bp->bbuf, (size_t)i);
			bp->rdline = i;
			bp->icount = -i;
			bp->gbuf = bp->ebuf-i;
			return 0;
		}
		bp->offset += j;
		i += j;
		ep = memchr(ip, delim, (size_t)j);
		if(ep) {
			/*
			 * found in new piece
			 * copy back up and reset everything
			 */
			ip = (char*)bp->ebuf - i;
			if(i < bp->bsize){
				memmove(ip, bp->bbuf, (size_t)i);
				bp->gbuf = (unsigned char*)ip;
			}
			j = (int)((ep - (char*)bp->bbuf) + 1);
			bp->rdline = j;
			bp->icount = j - i;
			return ip;
		}
		ip += j;
	}

	/*
	 * full buffer without finding
	 */
	bp->rdline = bp->bsize;
	bp->icount = -bp->bsize;
	bp->gbuf = bp->bbuf;
	return 0;
}

int
Blinelen(Biobuf *bp)
{

	return bp->rdline;
}
