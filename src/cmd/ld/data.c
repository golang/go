// Inferno utils/8l/asm.c
// http://code.google.com/p/inferno-os/source/browse/utils/8l/asm.c
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

// Data layout and relocation.

#include	"l.h"
#include	"../ld/lib.h"
#include	"../ld/elf.h"

/*
 * divide-and-conquer list-link
 * sort of Sym* structures.
 * Used for the data block.
 */
int
datcmp(Sym *s1, Sym *s2)
{
	if(s1->type != s2->type)
		return (int)s1->type - (int)s2->type;
	if(s1->size != s2->size) {
		if(s1->size < s2->size)
			return -1;
		return +1;
	}
	return strcmp(s1->name, s2->name);
}

Sym*
datsort(Sym *l)
{
	Sym *l1, *l2, *le;

	if(l == 0 || l->next == 0)
		return l;

	l1 = l;
	l2 = l;
	for(;;) {
		l2 = l2->next;
		if(l2 == 0)
			break;
		l2 = l2->next;
		if(l2 == 0)
			break;
		l1 = l1->next;
	}

	l2 = l1->next;
	l1->next = 0;
	l1 = datsort(l);
	l2 = datsort(l2);

	/* set up lead element */
	if(datcmp(l1, l2) < 0) {
		l = l1;
		l1 = l1->next;
	} else {
		l = l2;
		l2 = l2->next;
	}
	le = l;

	for(;;) {
		if(l1 == 0) {
			while(l2) {
				le->next = l2;
				le = l2;
				l2 = l2->next;
			}
			le->next = 0;
			break;
		}
		if(l2 == 0) {
			while(l1) {
				le->next = l1;
				le = l1;
				l1 = l1->next;
			}
			break;
		}
		if(datcmp(l1, l2) < 0) {
			le->next = l1;
			le = l1;
			l1 = l1->next;
		} else {
			le->next = l2;
			le = l2;
			l2 = l2->next;
		}
	}
	le->next = 0;
	return l;
}

Reloc*
addrel(Sym *s)
{
	if(s->nr >= s->maxr) {
		if(s->maxr == 0)
			s->maxr = 4;
		else
			s->maxr <<= 1;
		s->r = realloc(s->r, s->maxr*sizeof s->r[0]);
		if(s->r == 0) {
			diag("out of memory");
			errorexit();
		}
		memset(s->r+s->nr, 0, (s->maxr-s->nr)*sizeof s->r[0]);
	}
	return &s->r[s->nr++];
}

void
relocsym(Sym *s)
{
	Reloc *r;
	Prog p;
	int32 i, off, siz, fl;
	vlong o;
	uchar *cast;
	
	memset(&p, 0, sizeof p);
	for(r=s->r; r<s->r+s->nr; r++) {
		off = r->off;
		siz = r->siz;
		switch(r->type) {
		default:
			diag("unknown reloc %d", r->type);
		case D_ADDR:
			o = symaddr(r->sym);
			break;
		case D_SIZE:
			o = r->sym->size;
			break;
		}
		o += r->add;
		switch(siz) {
		default:
			diag("bad reloc size %d", siz);
		case 4:
			fl = o;
			cast = (uchar*)&fl;
			for(i=0; i<4; i++)
				s->p[off+i] = cast[inuxi4[i]];
			break;
		case 8:
			cast = (uchar*)&o;
			for(i=0; i<8; i++)
				s->p[off+i] = cast[inuxi8[i]];
			break;
		}		
	}
}

void
reloc(void)
{
	Sym *s;
	
	if(debug['v'])
		Bprint(&bso, "%5.2f reloc\n", cputime());
	Bflush(&bso);

	for(s=textp; s!=S; s=s->next)
		relocsym(s);
	for(s=datap; s!=S; s=s->next) {
		if(!s->reachable)
			diag("unerachable? %s", s->name);
		relocsym(s);
	}
}

void
symgrow(Sym *s, int32 siz)
{
	if(s->np >= siz)
		return;

	if(s->maxp < siz) {
		if(s->maxp == 0)
			s->maxp = 8;
		while(s->maxp < siz)
			s->maxp <<= 1;
		s->p = realloc(s->p, s->maxp);
		if(s->p == nil) {
			diag("out of memory");
			errorexit();
		}
		memset(s->p+s->np, 0, s->maxp-s->np);
	}
	s->np = siz;
}

void
savedata(Sym *s, Prog *p)
{
	int32 off, siz, i, fl;
	uchar *cast;
	vlong o;
	Reloc *r;
	
	off = p->from.offset;
	siz = p->datasize;
	symgrow(s, off+siz);
	
	switch(p->to.type) {
	default:
		diag("bad data: %P", p);
		break;

	case D_FCONST:
		switch(siz) {
		default:
		case 4:
			fl = ieeedtof(&p->to.ieee);
			cast = (uchar*)&fl;
			for(i=0; i<4; i++)
				s->p[off+i] = cast[fnuxi4[i]];
			break;
		case 8:
			cast = (uchar*)&p->to.ieee;
			for(i=0; i<8; i++)
				s->p[off+i] = cast[fnuxi8[i]];
			break;
		}
		break;
	
	case D_SCONST:
		for(i=0; i<siz; i++)
			s->p[off+i] = p->to.scon[i];
		break;
	
	case D_CONST:
		if(p->to.sym)
			goto Addr;
		o = p->to.offset;
		fl = o;
		cast = (uchar*)&fl;
		switch(siz) {
		default:
			diag("bad nuxi %d\n%P", siz, p);
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
		break;

	case D_ADDR:
	case D_SIZE:
	Addr:
		r = addrel(s);
		r->off = off;
		r->siz = siz;
		r->sym = p->to.sym;
		r->type = p->to.type;
		if(r->type != D_SIZE)
			r->type = D_ADDR;
		r->add = p->to.offset;
		break;
	}
}

static void
blk(Sym *allsym, int32 addr, int32 size)
{
	Sym *sym;
	int32 eaddr;
	uchar *p, *ep;

	for(sym = allsym; sym != nil; sym = sym->next)
		if(sym->value >= addr)
			break;

	eaddr = addr+size;
	for(; sym != nil; sym = sym->next) {
		if(sym->value >= eaddr)
			break;
		if(sym->value < addr) {
			diag("phase error: addr=%#llx but sym=%#llx type=%d", addr, sym->value, sym->type);
			errorexit();
		}
		cursym = sym;
		for(; addr < sym->value; addr++)
			cput(0);
		p = sym->p;
		ep = p + sym->np;
		while(p < ep)
			cput(*p++);
		addr += sym->np;
		for(; addr < sym->value+sym->size; addr++)
			cput(0);
		if(addr != sym->value+sym->size) {
			diag("phase error: addr=%#llx value+size=%#llx", addr, sym->value+sym->size);
			errorexit();
		}
	}
	
	for(; addr < eaddr; addr++)
		cput(0);
	cflush();
}
			
void
datblk(int32 addr, int32 size)
{
	Sym *sym;
	int32 eaddr;
	uchar *p, *ep;

	blk(datap, addr, size);

	/* again for printing */
	if(!debug['a'])
		return;

	for(sym = datap; sym != nil; sym = sym->next)
		if(sym->value >= addr)
			break;

	eaddr = addr + size;
	for(; sym != nil; sym = sym->next) {
		if(sym->value >= eaddr)
			break;
		if(addr < sym->value) {
			Bprint(&bso, "%-20s %.8ux|", "(pre-pad)", addr);
			for(; addr < sym->value; addr++)
				Bprint(&bso, " %.2ux", 0);
			Bprint(&bso, "\n");
		}
		Bprint(&bso, "%-20s %.8ux|", sym->name, addr);
		p = sym->p;
		ep = p + sym->np;
		while(p < ep)
			Bprint(&bso, " %.2ux", *p++);
		addr += sym->np;
		for(; addr < sym->value+sym->size; addr++)
			Bprint(&bso, " %.2ux", 0);
		Bprint(&bso, "\n");
	}

	if(addr < eaddr) {
		Bprint(&bso, "%-20s %.8ux|", "(post-pad)", addr);
		for(; addr < eaddr; addr++)
			Bprint(&bso, " %.2ux", 0);
	}
}

void
strnput(char *s, int n)
{
	for(; *s && n > 0; s++) {
		cput(*s);
		n--;
	}
	while(n > 0) {
		cput(0);
		n--;
	}
}

vlong
addstring(Sym *s, char *str)
{
	int n;
	int32 r;

	if(s->type == 0)
		s->type = SDATA;
	s->reachable = 1;
	r = s->size;
	n = strlen(str)+1;
	if(strcmp(s->name, ".shstrtab") == 0)
		elfsetstring(str, r);
	symgrow(s, r+n);
	memmove(s->p+r, str, n);
	s->size += n;
	return r;
}

vlong
adduintxx(Sym *s, uint64 v, int wid)
{
	int32 i, r, fl;
	vlong o;
	uchar *cast;

	if(s->type == 0)
		s->type = SDATA;
	s->reachable = 1;
	r = s->size;
	s->size += wid;
	symgrow(s, s->size);
	fl = v;
	cast = (uchar*)&fl;
	switch(wid) {
	case 1:
		s->p[r] = cast[inuxi1[0]];
		break;
	case 2:
		for(i=0; i<2; i++)
			s->p[r+i] = cast[inuxi2[i]];
		break;
	case 4:
		for(i=0; i<4; i++)
			s->p[r+i] = cast[inuxi4[i]];
		break;
	case 8:
		o = v;
		cast = (uchar*)&o;
		for(i=0; i<8; i++)
			s->p[r+i] = cast[inuxi8[i]];
		break;
	}
	return r;
}

vlong
adduint8(Sym *s, uint8 v)
{
	return adduintxx(s, v, 1);
}

vlong
adduint16(Sym *s, uint16 v)
{
	return adduintxx(s, v, 2);
}

vlong
adduint32(Sym *s, uint32 v)
{
	return adduintxx(s, v, 4);
}

vlong
adduint64(Sym *s, uint64 v)
{
	return adduintxx(s, v, 8);
}

vlong
addaddr(Sym *s, Sym *t)
{
	vlong i;
	Reloc *r;

	if(s->type == 0)
		s->type = SDATA;
	s->reachable = 1;
	i = s->size;
	s->size += PtrSize;
	symgrow(s, s->size);
	r = addrel(s);
	r->sym = t;
	r->off = i;
	r->siz = PtrSize;
	r->type = D_ADDR;
	return i;
}

vlong
addsize(Sym *s, Sym *t)
{
	vlong i;
	Reloc *r;

	if(s->type == 0)
		s->type = SDATA;
	s->reachable = 1;
	i = s->size;
	s->size += PtrSize;
	symgrow(s, s->size);
	r = addrel(s);
	r->sym = t;
	r->off = i;
	r->siz = PtrSize;
	r->type = D_SIZE;
	return i;
}

void
dodata(void)
{
	int32 h, t, datsize;
	Section *sect;
	Sym *s, *last;

	if(debug['v'])
		Bprint(&bso, "%5.2f dodata\n", cputime());
	Bflush(&bso);

	segdata.rwx = 06;
	segdata.vaddr = 0;	/* span will += INITDAT */

	last = nil;
	datap = nil;
	for(h=0; h<NHASH; h++) {
		for(s=hash[h]; s!=S; s=s->hash){
			if(!s->reachable)
				continue;
			if(STEXT < s->type && s->type < SXREF) {
				if(last == nil)
					datap = s;
				else
					last->next = s;
				s->next = nil;
				last = s;
			}
		}
	}

	for(s = datap; s != nil; s = s->next) {
		if(s->np > 0 && s->type == SBSS)	// TODO: necessary?
			s->type = SDATA;
		if(s->np > s->size)
			diag("%s: initialize bounds (%lld < %d)",
				s->name, s->size, s->np);
	}
	datap = datsort(datap);

	/*
	 * allocate data sections.  list is sorted by type,
	 * so we can just walk it for each piece we want to emit.
	 */

	sect = addsection(&segtext, ".text", 05);	// set up for span TODO(rsc): clumsy
	
	/* read-only data */
	sect = addsection(&segtext, ".rodata", 06);
	sect->vaddr = 0;
	datsize = 0;
	s = datap;
	for(; s != nil && s->type < SDATA; s = s->next) {
		s->type = SRODATA;
		t = rnd(s->size, 4);
		s->size = t;
		s->value = datsize;
		datsize += t;
	}
	sect->len = datsize - sect->vaddr;
	
	/* data */
	datsize = 0;
	sect = addsection(&segdata, ".data", 06);
	sect->vaddr = 0;
	for(; s != nil && s->type < SBSS; s = s->next) {
		s->type = SDATA;
		t = s->size;
		if(t == 0 && s->name[0] != '.') {
			diag("%s: no size", s->name);
			t = 1;
		}
		if(t & 1)
			;
		else if(t & 2)
			datsize = rnd(datsize, 2);
		else if(t & 4)
			datsize = rnd(datsize, 4);
		else
			datsize = rnd(datsize, 8);
		s->value = datsize;
		datsize += t;
	}
	sect->len = datsize - sect->vaddr;
	segdata.filelen = datsize;

	/* bss */
	sect = addsection(&segdata, ".bss", 06);
	sect->vaddr = datsize;
	for(; s != nil; s = s->next) {
		if(s->type != SBSS) {
			cursym = s;
			diag("unexpected symbol type %d", s->type);
		}
		t = s->size;
		if(t & 1)
			;
		else if(t & 2)
			datsize = rnd(datsize, 2);
		else if(t & 4)
			datsize = rnd(datsize, 4);
		else
			datsize = rnd(datsize, 8);
		s->size = t;
		s->value = datsize;
		datsize += t;
	}
	sect->len = datsize - sect->vaddr;
	segdata.len = datsize;

	xdefine("data", SBSS, 0);
	xdefine("edata", SBSS, segdata.filelen);
	xdefine("end", SBSS, segdata.len);

	if(debug['s'] || HEADTYPE == 8)
		xdefine("symdat", SFIXED, 0);
	else
		xdefine("symdat", SFIXED, SYMDATVA);
}

