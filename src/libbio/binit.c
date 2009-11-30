/*
http://code.google.com/p/inferno-os/source/browse/libbio/binit.c

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

enum
{
	MAXBUFS	= 20
};

static	Biobuf*	wbufs[MAXBUFS];
static	int		atexitflag;

static
void
batexit(void)
{
	Biobuf *bp;
	int i;

	for(i=0; i<MAXBUFS; i++) {
		bp = wbufs[i];
		if(bp != 0) {
			wbufs[i] = 0;
			Bflush(bp);
		}
	}
}

static
void
deinstall(Biobuf *bp)
{
	int i;

	for(i=0; i<MAXBUFS; i++)
		if(wbufs[i] == bp)
			wbufs[i] = 0;
}

static
void
install(Biobuf *bp)
{
	int i;

	deinstall(bp);
	for(i=0; i<MAXBUFS; i++)
		if(wbufs[i] == 0) {
			wbufs[i] = bp;
			break;
		}
	if(atexitflag == 0) {
		atexitflag = 1;
		atexit(batexit);
	}
}

int
Binits(Biobuf *bp, int f, int mode, unsigned char *p, int size)
{

	p += Bungetsize;	/* make room for Bungets */
	size -= Bungetsize;

	switch(mode&~(ORCLOSE|OTRUNC)) {
	default:
		fprint(2, "Bopen: unknown mode %d\n", mode);
		return Beof;

	case OREAD:
		bp->state = Bractive;
		bp->ocount = 0;
		break;

	case OWRITE:
		install(bp);
		bp->state = Bwactive;
		bp->ocount = -size;
		break;
	}
	bp->bbuf = p;
	bp->ebuf = p+size;
	bp->bsize = size;
	bp->icount = 0;
	bp->gbuf = bp->ebuf;
	bp->fid = f;
	bp->flag = 0;
	bp->rdline = 0;
	bp->offset = 0;
	bp->runesize = 0;
	return 0;
}


int
Binit(Biobuf *bp, int f, int mode)
{
	return Binits(bp, f, mode, bp->b, sizeof(bp->b));
}

Biobuf*
Bfdopen(int f, int mode)
{
	Biobuf *bp;

	bp = malloc(sizeof(Biobuf));
	if(bp == 0)
		return 0;
	Binits(bp, f, mode, bp->b, sizeof(bp->b));
	bp->flag = Bmagic;
	return bp;
}

Biobuf*
Bopen(char *name, int mode)
{
	Biobuf *bp;
	int f;

	switch(mode&~(ORCLOSE|OTRUNC)) {
	default:
		fprint(2, "Bopen: unknown mode %d\n", mode);
		return 0;

	case OREAD:
		f = open(name, OREAD);
		if(f < 0)
			return 0;
		break;

	case OWRITE:
		f = create(name, OWRITE|OTRUNC, 0666);
		if(f < 0)
			return 0;
	}
	bp = Bfdopen(f, mode);
	if(bp == 0)
		close(f);
	return bp;
}

int
Bterm(Biobuf *bp)
{

	deinstall(bp);
	Bflush(bp);
	if(bp->flag == Bmagic) {
		bp->flag = 0;
		close(bp->fid);
		free(bp);
	}
	return 0;
}
