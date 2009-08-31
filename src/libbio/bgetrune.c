/*
http://code.google.com/p/inferno-os/source/browse/libbio/bgetrune.c

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
#include	<utf.h>

long
Bgetrune(Biobuf *bp)
{
	int c, i;
	Rune rune;
	char str[UTFmax];

	c = Bgetc(bp);
	if(c < Runeself) {		/* one char */
		bp->runesize = 1;
		return c;
	}
	str[0] = c;

	for(i=1;;) {
		c = Bgetc(bp);
		if(c < 0)
			return c;
		str[i++] = c;

		if(fullrune(str, i)) {
			bp->runesize = chartorune(&rune, str);
			while(i > bp->runesize) {
				Bungetc(bp);
				i--;
			}
			return rune;
		}
	}
}

int
Bungetrune(Biobuf *bp)
{

	if(bp->state == Bracteof)
		bp->state = Bractive;
	if(bp->state != Bractive)
		return Beof;
	bp->icount -= bp->runesize;
	bp->runesize = 0;
	return 1;
}
