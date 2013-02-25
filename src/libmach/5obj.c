// Inferno libmach/5obj.c
// http://code.google.com/p/inferno-os/source/browse/utils/libmach/5obj.c
//
// 	Copyright © 1994-1999 Lucent Technologies Inc.
// 	Power PC support Copyright © 1995-2004 C H Forsyth (forsyth@terzarima.net).
// 	Portions Copyright © 1997-1999 Vita Nuova Limited.
// 	Portions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com).
// 	Revisions Copyright © 2000-2004 Lucent Technologies Inc. and others.
//	Portions Copyright © 2009 The Go Authors.  All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

/*
 * 5obj.c - identify and parse an arm object file
 */
#include <u.h>
#include <libc.h>
#include <bio.h>
#include <mach.h>
#include "../cmd/5l/5.out.h"
#include "obj.h"

typedef struct Addr	Addr;
struct Addr
{
	char	type;
	char	sym;
	char	name;
	char	gotype;
};
static Addr addr(Biobuf*);
static char type2char(int);
static void skip(Biobuf*, int);

int
_is5(char *s)
{
	return  s[0] == ANAME				/* ANAME */
		&& s[1] == D_FILE			/* type */
		&& s[2] == 1				/* sym */
		&& s[3] == '<';				/* name of file */
}

int
_read5(Biobuf *bp, Prog *p)
{
	int as, n;
	Addr a;

	as = Bgetc(bp);			/* as */
	if(as < 0)
		return 0;
	p->kind = aNone;
	p->sig = 0;
	if(as == ANAME || as == ASIGNAME){
		if(as == ASIGNAME){
			Bread(bp, &p->sig, 4);
			p->sig = leswal(p->sig);
		}
		p->kind = aName;
		p->type = type2char(Bgetc(bp));		/* type */
		p->sym = Bgetc(bp);			/* sym */
		n = 0;
		for(;;) {
			as = Bgetc(bp);
			if(as < 0)
				return 0;
			n++;
			if(as == 0)
				break;
		}
		p->id = malloc(n);
		if(p->id == 0)
			return 0;
		Bseek(bp, -n, 1);
		if(Bread(bp, p->id, n) != n)
			return 0;
		return 1;
	}
	if(as == ATEXT)
		p->kind = aText;
	else if(as == AGLOBL)
		p->kind = aData;
	skip(bp, 6);		/* scond(1), reg(1), lineno(4) */
	a = addr(bp);
	addr(bp);
	if(a.type != D_OREG || a.name != D_STATIC && a.name != D_EXTERN)
		p->kind = aNone;
	p->sym = a.sym;
	return 1;
}

static Addr
addr(Biobuf *bp)
{
	Addr a;
	long off;

	a.type = Bgetc(bp);	/* a.type */
	skip(bp,1);		/* reg */
	a.sym = Bgetc(bp);	/* sym index */
	a.name = Bgetc(bp);	/* sym type */
	a.gotype = Bgetc(bp);	/* go type */
	switch(a.type){
	default:
	case D_NONE:
	case D_REG:
	case D_FREG:
	case D_PSR:
	case D_FPCR:
		break;
	case D_REGREG:
	case D_REGREG2:
		Bgetc(bp);
		break;
	case D_CONST2:
		Bgetc(bp);
		Bgetc(bp);
		Bgetc(bp);
		Bgetc(bp);	// fall through
	case D_OREG:
	case D_CONST:
	case D_BRANCH:
	case D_SHIFT:
		off = Bgetc(bp);
		off |= Bgetc(bp) << 8;
		off |= Bgetc(bp) << 16;
		off |= Bgetc(bp) << 24;
		if(off < 0)
			off = -off;
		if(a.sym && (a.name==D_PARAM || a.name==D_AUTO))
			_offset(a.sym, off);
		break;
	case D_SCONST:
		skip(bp, NSNAME);
		break;
	case D_FCONST:
		skip(bp, 8);
		break;
	}
	return a;
}

static char
type2char(int t)
{
	switch(t){
	case D_EXTERN:		return 'U';
	case D_STATIC:		return 'b';
	case D_AUTO:		return 'a';
	case D_PARAM:		return 'p';
	default:		return UNKNOWN;
	}
}

static void
skip(Biobuf *bp, int n)
{
	while (n-- > 0)
		Bgetc(bp);
}
