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
		case D_PCREL:
			o = symaddr(r->sym) - (s->value + r->off + r->siz);
			break;
		case D_SIZE:
			o = r->sym->size;
			break;
		}
		o += r->add;
		switch(siz) {
		default:
			diag("bad reloc size %#ux", siz);
		case 4 + Rbig:
			fl = o;
			s->p[off] = fl>>24;
			s->p[off+1] = fl>>16;
			s->p[off+2] = fl>>8;
			s->p[off+3] = fl;
			break;
		case 4 + Rlittle:
			fl = o;
			s->p[off] = fl;
			s->p[off+1] = fl>>8;
			s->p[off+2] = fl>>16;
			s->p[off+3] = fl>>24;
			break;
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
	for(s=datap; s!=S; s=s->next)
		relocsym(s);
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
			diag("phase error: addr=%#llx but sym=%#llx type=%d", (vlong)addr, (vlong)sym->value, sym->type);
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
codeblk(int32 addr, int32 size)
{
	Sym *sym;
	int32 eaddr, n, epc;
	Prog *p;
	uchar *q;

	if(debug['a'])
		Bprint(&bso, "codeblk [%#x,%#x) at offset %#llx\n", addr, addr+size, seek(cout, 0, 1));

	blk(textp, addr, size);

	/* again for printing */
	if(!debug['a'])
		return;

	for(sym = textp; sym != nil; sym = sym->next) {
		if(!sym->reachable)
			continue;
		if(sym->value >= addr)
			break;
	}

	eaddr = addr + size;
	for(; sym != nil; sym = sym->next) {
		if(!sym->reachable)
			continue;
		if(sym->value >= eaddr)
			break;

		if(addr < sym->value) {
			Bprint(&bso, "%-20s %.8llux|", "_", (vlong)addr);
			for(; addr < sym->value; addr++)
				Bprint(&bso, " %.2ux", 0);
			Bprint(&bso, "\n");
		}
		p = sym->text;
		Bprint(&bso, "%.6llux\t%-20s | %P\n", (vlong)addr, sym->name, p);
		for(p = p->link; p != P; p = p->link) {
			if(p->link != P)
				epc = p->link->pc;
			else
				epc = sym->value + sym->size;
			Bprint(&bso, "%.6ux\t", p->pc);
			q = sym->p + p->pc - sym->value;
			n = epc - p->pc;
			Bprint(&bso, "%-20.*I | %P\n", n, q, p);
			addr += n;
		}
	}

	if(addr < eaddr) {
		Bprint(&bso, "%-20s %.8llux|", "_", (vlong)addr);
		for(; addr < eaddr; addr++)
			Bprint(&bso, " %.2ux", 0);
	}
	Bflush(&bso);
}
			
void
datblk(int32 addr, int32 size)
{
	Sym *sym;
	int32 eaddr;
	uchar *p, *ep;

	if(debug['a'])
		Bprint(&bso, "datblk [%#x,%#x) at offset %#llx\n", addr, addr+size, seek(cout, 0, 1));

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
			Bprint(&bso, "%-20s %.8ux| 00 ...\n", "(pre-pad)", addr);
			addr = sym->value;
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

	if(addr < eaddr)
		Bprint(&bso, "%-20s %.8ux| 00 ...\n", "(post-pad)", addr);
	Bprint(&bso, "%-20s %.8ux|\n", "", eaddr);
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

	last = nil;
	datap = nil;
	for(h=0; h<NHASH; h++) {
		for(s=hash[h]; s!=S; s=s->hash){
			if(!s->reachable || s->special)
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
	datsize += dynptrsize;

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
}

// assign addresses
void
address(void)
{
	Section *s, *text, *data, *rodata, *bss;
	Sym *sym;
	uvlong va;

	va = INITTEXT;
	segtext.rwx = 05;
	segtext.vaddr = va;
	segtext.fileoff = HEADR;
	for(s=segtext.sect; s != nil; s=s->next) {
		s->vaddr = va;
		va += s->len;
		segtext.len = va - INITTEXT;
		va = rnd(va, INITRND);
	}
	segtext.filelen = segtext.len;

	segdata.rwx = 06;
	segdata.vaddr = va;
	segdata.fileoff = va - segtext.vaddr + segtext.fileoff;
	for(s=segdata.sect; s != nil; s=s->next) {
		s->vaddr = va;
		va += s->len;
		if(s == segdata.sect)
			va += dynptrsize;
		segdata.len = va - segdata.vaddr;
	}
	segdata.filelen = segdata.sect->len + dynptrsize;	// assume .data is first
	
	text = segtext.sect;
	rodata = segtext.sect->next;
	data = segdata.sect;
	bss = segdata.sect->next;

	for(sym = datap; sym != nil; sym = sym->next) {
		cursym = sym;
		if(sym->type < SDATA)
			sym->value += rodata->vaddr;
		else
			sym->value += data->vaddr;
	}
	
	xdefine("text", STEXT, text->vaddr);
	xdefine("etext", STEXT, text->vaddr + text->len);
	xdefine("rodata", SRODATA, rodata->vaddr);
	xdefine("erodata", SRODATA, rodata->vaddr + rodata->len);
	xdefine("data", SBSS, data->vaddr);
	xdefine("edata", SBSS, data->vaddr + data->len);
	xdefine("end", SBSS, segdata.vaddr + segdata.len);

	sym = lookup("pclntab", 0);
	xdefine("epclntab", SRODATA, sym->value + sym->size);
	sym = lookup("symtab", 0);
	xdefine("esymtab", SRODATA, sym->value + sym->size);
}
