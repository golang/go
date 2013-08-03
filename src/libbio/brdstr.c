/*
	Copyright © 2009 The Go Authors.  All rights reserved.

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

char*
Brdstr(Biobuf *bp, int delim, int nulldelim)
{
	char *p, *q, *nq;
	int n, linelen;

	q = nil;
	n = 0;
	for(;;) {
		p = Brdline(bp, delim);
		linelen = Blinelen(bp);
		if(n == 0 && linelen == 0)
			return nil;
		nq = realloc(q, (size_t)(n+linelen+1));
		if(nq == nil) {
			free(q);
			return nil;
		}
		q = nq;
		if(p != nil) {
			memmove(q+n, p, (size_t)linelen);
			n += linelen;
			if(nulldelim)
				q[n-1] = '\0';
			break;
		}
		if(linelen == 0)
			break;
		Bread(bp, q+n, linelen);
		n += linelen;
	}
	q[n] = '\0';
	return q;
}
