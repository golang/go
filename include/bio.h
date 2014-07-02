/*
http://code.google.com/p/inferno-os/source/browse/include/bio.h

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

#ifndef _BIO_H_
#define _BIO_H_ 1
#if defined(__cplusplus)
extern "C" {
#endif

#ifdef AUTOLIB
AUTOLIB(bio)
#endif

typedef	struct	Biobuf	Biobuf;

enum
{
	Bsize		= 8*1024,
	Bungetsize	= 4,		/* space for ungetc */
	Bmagic		= 0x314159,
	Beof		= -1,
	Bbad		= -2,

	Binactive	= 0,		/* states */
	Bractive,
	Bwactive,
	Bracteof,

	Bend
};

struct	Biobuf
{
	int	icount;		/* neg num of bytes at eob */
	int	ocount;		/* num of bytes at bob */
	int	rdline;		/* num of bytes after rdline */
	int	runesize;	/* num of bytes of last getrune */
	int	state;		/* r/w/inactive */
	int	fid;		/* open file */
	int	flag;		/* magic if malloc'ed */
	vlong	offset;		/* offset of buffer in file */
	int	bsize;		/* size of buffer */
	unsigned char*	bbuf;		/* pointer to beginning of buffer */
	unsigned char*	ebuf;		/* pointer to end of buffer */
	unsigned char*	gbuf;		/* pointer to good data in buf */
	unsigned char	b[Bungetsize+Bsize];
};

/*
 * These macros get 1-, 2-, and 4-byte integer values by reading the
 * next few bytes in little-endian order.
 */
#define	BGETC(bp)\
	((bp)->icount?(int)((bp)->ebuf[(bp)->icount++]):Bgetc((bp)))
#define	BGETLE2(bp)\
	((bp)->icount<=-2?((bp)->icount+=2,((bp)->ebuf[(bp)->icount-2])|((bp)->ebuf[(bp)->icount-1]<<8)):Bgetle2((bp)))
#define	BGETLE4(bp)\
	(int)((bp)->icount<=-4?((bp)->icount+=4,((bp)->ebuf[(bp)->icount-4])|((bp)->ebuf[(bp)->icount-3]<<8)|((bp)->ebuf[(bp)->icount-2]<<16)|((uint32)(bp)->ebuf[(bp)->icount-1]<<24)):Bgetle4((bp)))

/*
 * These macros put 1-, 2-, and 4-byte integer values by writing the
 * next few bytes in little-endian order.
 */
#define	BPUTC(bp,c)\
	((bp)->ocount?(bp)->ebuf[(bp)->ocount++]=(unsigned char)(c),0:Bputc((bp),(c)))
#define	BPUTLE2(bp,c)\
	((bp)->ocount<=-2?(bp)->ocount+=2,(bp)->ebuf[(bp)->ocount-2]=(unsigned char)(c),(bp)->ebuf[(bp)->ocount-1]=(unsigned char)(c>>8),0:Bputle2((bp),(c)))
#define	BPUTLE4(bp,c)\
	((bp)->ocount<=-4?(bp)->ocount+=4,(bp)->ebuf[(bp)->ocount-4]=(unsigned char)(c),(bp)->ebuf[(bp)->ocount-3]=(unsigned char)(c>>8),(bp)->ebuf[(bp)->ocount-2]=(unsigned char)(c>>16),(bp)->ebuf[(bp)->ocount-1]=(unsigned char)(c>>24),0:Bputle4((bp),(c)))

#define	BOFFSET(bp)\
	(((bp)->state==Bractive)?\
		(bp)->offset + (bp)->icount:\
	(((bp)->state==Bwactive)?\
		(bp)->offset + ((bp)->bsize + (bp)->ocount):\
		-1))
#define	BLINELEN(bp)\
	(bp)->rdline
#define	BFILDES(bp)\
	(bp)->fid

int	Bbuffered(Biobuf*);
Biobuf*	Bfdopen(int, int);
int	Bfildes(Biobuf*);
int	Bflush(Biobuf*);
int	Bgetc(Biobuf*);
int	Bgetle2(Biobuf*);
int	Bgetle4(Biobuf*);
int	Bgetd(Biobuf*, double*);
long	Bgetrune(Biobuf*);
int	Binit(Biobuf*, int, int);
int	Binits(Biobuf*, int, int, unsigned char*, int);
int	Blinelen(Biobuf*);
vlong	Boffset(Biobuf*);
Biobuf*	Bopen(char*, int);
int	Bprint(Biobuf*, char*, ...);
int	Bputc(Biobuf*, int);
int	Bputle2(Biobuf*, int);
int	Bputle4(Biobuf*, int);
int	Bputrune(Biobuf*, long);
void*	Brdline(Biobuf*, int);
char*	Brdstr(Biobuf*, int, int);
long	Bread(Biobuf*, void*, long);
vlong	Bseek(Biobuf*, vlong, int);
int	Bterm(Biobuf*);
int	Bungetc(Biobuf*);
int	Bungetrune(Biobuf*);
long	Bwrite(Biobuf*, void*, long);
int	Bvprint(Biobuf*, char*, va_list);
/*c2go
int	BGETC(Biobuf*);
int	BGETLE2(Biobuf*);
int	BGETLE4(Biobuf*);
int	BPUTC(Biobuf*, int);
int	BPUTLE2(Biobuf*, int);
int	BPUTLE4(Biobuf*, int);
*/

#if defined(__cplusplus)
}
#endif
#endif
