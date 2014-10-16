// Derived from Inferno utils/6l/obj.c and utils/6l/span.c
// http://code.google.com/p/inferno-os/source/browse/utils/6l/obj.c
// http://code.google.com/p/inferno-os/source/browse/utils/6l/span.c
//
//	Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
//	Portions Copyright © 1995-1997 C H Forsyth (forsyth@terzarima.net)
//	Portions Copyright © 1997-1999 Vita Nuova Limited
//	Portions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com)
//	Portions Copyright © 2004,2006 Bruce Ellis
//	Portions Copyright © 2005-2007 C H Forsyth (forsyth@terzarima.net)
//	Revisions Copyright © 2000-2007 Lucent Technologies Inc. and others
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

#include <u.h>
#include <libc.h>
#include <bio.h>
#include <link.h>

void
mangle(char *file)
{
	sysfatal("%s: mangled input file", file);
}

void
symgrow(Link *ctxt, LSym *s, vlong lsiz)
{
	int32 siz;

	USED(ctxt);

	siz = (int32)lsiz;
	if((vlong)siz != lsiz)
		sysfatal("symgrow size %lld too long", lsiz);

	if(s->np >= siz)
		return;

	if(s->np > s->maxp) {
		ctxt->cursym = s;
		sysfatal("corrupt symbol data: np=%lld > maxp=%lld", (vlong)s->np, (vlong)s->maxp);
	}

	if(s->maxp < siz) {
		if(s->maxp == 0)
			s->maxp = 8;
		while(s->maxp < siz)
			s->maxp <<= 1;
		s->p = erealloc(s->p, s->maxp);
		memset(s->p+s->np, 0, s->maxp-s->np);
	}
	s->np = siz;
}

void
savedata(Link *ctxt, LSym *s, Prog *p, char *pn)
{
	int32 off, siz, i, fl;
	float32 flt;
	uchar *cast;
	vlong o;
	Reloc *r;

	off = p->from.offset;
	siz = ctxt->arch->datasize(p);
	if(off < 0 || siz < 0 || off >= 1<<30 || siz >= 100)
		mangle(pn);
	if(ctxt->enforce_data_order && off < s->np)
		ctxt->diag("data out of order (already have %d)\n%P", p);
	symgrow(ctxt, s, off+siz);

	if(p->to.type == ctxt->arch->D_FCONST) {
		switch(siz) {
		default:
		case 4:
			flt = p->to.u.dval;
			cast = (uchar*)&flt;
			for(i=0; i<4; i++)
				s->p[off+i] = cast[fnuxi4[i]];
			break;
		case 8:
			cast = (uchar*)&p->to.u.dval;
			for(i=0; i<8; i++)
				s->p[off+i] = cast[fnuxi8[i]];
			break;
		}
	} else if(p->to.type == ctxt->arch->D_SCONST) {
		for(i=0; i<siz; i++)
			s->p[off+i] = p->to.u.sval[i];
	} else if(p->to.type == ctxt->arch->D_CONST) {
		if(p->to.sym)
			goto addr;
		o = p->to.offset;
		fl = o;
		cast = (uchar*)&fl;
		switch(siz) {
		default:
			ctxt->diag("bad nuxi %d\n%P", siz, p);
			break;
		case 1:
			s->p[off] = cast[inuxi1[0]];
			break;
		case 2:
			for(i=0; i<2; i++)
				s->p[off+i] = cast[inuxi2[i]];
			break;
		case 4:
			for(i=0; i<4; i++)
				s->p[off+i] = cast[inuxi4[i]];
			break;
		case 8:
			cast = (uchar*)&o;
			for(i=0; i<8; i++)
				s->p[off+i] = cast[inuxi8[i]];
			break;
		}
	} else if(p->to.type == ctxt->arch->D_ADDR) {
	addr:
		r = addrel(s);
		r->off = off;
		r->siz = siz;
		r->sym = p->to.sym;
		r->type = R_ADDR;
		r->add = p->to.offset;
	} else {
		ctxt->diag("bad data: %P", p);
	}
}

Reloc*
addrel(LSym *s)
{
	if(s->nr >= s->maxr) {
		if(s->maxr == 0)
			s->maxr = 4;
		else
			s->maxr <<= 1;
		s->r = erealloc(s->r, s->maxr*sizeof s->r[0]);
		memset(s->r+s->nr, 0, (s->maxr-s->nr)*sizeof s->r[0]);
	}
	return &s->r[s->nr++];
}

vlong
setuintxx(Link *ctxt, LSym *s, vlong off, uint64 v, vlong wid)
{
	int32 i, fl;
	vlong o;
	uchar *cast;

	if(s->type == 0)
		s->type = SDATA;
	s->reachable = 1;
	if(s->size < off+wid) {
		s->size = off+wid;
		symgrow(ctxt, s, s->size);
	}
	fl = v;
	cast = (uchar*)&fl;
	switch(wid) {
	case 1:
		s->p[off] = cast[inuxi1[0]];
		break;
	case 2:
		for(i=0; i<2; i++)
			s->p[off+i] = cast[inuxi2[i]];
		break;
	case 4:
		for(i=0; i<4; i++)
			s->p[off+i] = cast[inuxi4[i]];
		break;
	case 8:
		o = v;
		cast = (uchar*)&o;
		for(i=0; i<8; i++)
			s->p[off+i] = cast[inuxi8[i]];
		break;
	}
	return off+wid;
}

vlong
adduintxx(Link *ctxt, LSym *s, uint64 v, int wid)
{
	vlong off;

	off = s->size;
	setuintxx(ctxt, s, off, v, wid);
	return off;
}

vlong
adduint8(Link *ctxt, LSym *s, uint8 v)
{
	return adduintxx(ctxt, s, v, 1);
}

vlong
adduint16(Link *ctxt, LSym *s, uint16 v)
{
	return adduintxx(ctxt, s, v, 2);
}

vlong
adduint32(Link *ctxt, LSym *s, uint32 v)
{
	return adduintxx(ctxt, s, v, 4);
}

vlong
adduint64(Link *ctxt, LSym *s, uint64 v)
{
	return adduintxx(ctxt, s, v, 8);
}

vlong
setuint8(Link *ctxt, LSym *s, vlong r, uint8 v)
{
	return setuintxx(ctxt, s, r, v, 1);
}

vlong
setuint16(Link *ctxt, LSym *s, vlong r, uint16 v)
{
	return setuintxx(ctxt, s, r, v, 2);
}

vlong
setuint32(Link *ctxt, LSym *s, vlong r, uint32 v)
{
	return setuintxx(ctxt, s, r, v, 4);
}

vlong
setuint64(Link *ctxt, LSym *s, vlong r, uint64 v)
{
	return setuintxx(ctxt, s, r, v, 8);
}

vlong
addaddrplus(Link *ctxt, LSym *s, LSym *t, vlong add)
{
	vlong i;
	Reloc *r;

	if(s->type == 0)
		s->type = SDATA;
	s->reachable = 1;
	i = s->size;
	s->size += ctxt->arch->ptrsize;
	symgrow(ctxt, s, s->size);
	r = addrel(s);
	r->sym = t;
	r->off = i;
	r->siz = ctxt->arch->ptrsize;
	r->type = R_ADDR;
	r->add = add;
	return i + r->siz;
}

vlong
addpcrelplus(Link *ctxt, LSym *s, LSym *t, vlong add)
{
	vlong i;
	Reloc *r;

	if(s->type == 0)
		s->type = SDATA;
	s->reachable = 1;
	i = s->size;
	s->size += 4;
	symgrow(ctxt, s, s->size);
	r = addrel(s);
	r->sym = t;
	r->off = i;
	r->add = add;
	r->type = R_PCREL;
	r->siz = 4;
	return i + r->siz;
}

vlong
addaddr(Link *ctxt, LSym *s, LSym *t)
{
	return addaddrplus(ctxt, s, t, 0);
}

vlong
setaddrplus(Link *ctxt, LSym *s, vlong off, LSym *t, vlong add)
{
	Reloc *r;

	if(s->type == 0)
		s->type = SDATA;
	s->reachable = 1;
	if(off+ctxt->arch->ptrsize > s->size) {
		s->size = off + ctxt->arch->ptrsize;
		symgrow(ctxt, s, s->size);
	}
	r = addrel(s);
	r->sym = t;
	r->off = off;
	r->siz = ctxt->arch->ptrsize;
	r->type = R_ADDR;
	r->add = add;
	return off + r->siz;
}

vlong
setaddr(Link *ctxt, LSym *s, vlong off, LSym *t)
{
	return setaddrplus(ctxt, s, off, t, 0);
}

vlong
addsize(Link *ctxt, LSym *s, LSym *t)
{
	vlong i;
	Reloc *r;

	if(s->type == 0)
		s->type = SDATA;
	s->reachable = 1;
	i = s->size;
	s->size += ctxt->arch->ptrsize;
	symgrow(ctxt, s, s->size);
	r = addrel(s);
	r->sym = t;
	r->off = i;
	r->siz = ctxt->arch->ptrsize;
	r->type = R_SIZE;
	return i + r->siz;
}

vlong
addaddrplus4(Link *ctxt, LSym *s, LSym *t, vlong add)
{
	vlong i;
	Reloc *r;

	if(s->type == 0)
		s->type = SDATA;
	s->reachable = 1;
	i = s->size;
	s->size += 4;
	symgrow(ctxt, s, s->size);
	r = addrel(s);
	r->sym = t;
	r->off = i;
	r->siz = 4;
	r->type = R_ADDR;
	r->add = add;
	return i + r->siz;
}
