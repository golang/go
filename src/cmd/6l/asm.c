// Inferno utils/6l/asm.c
// http://code.google.com/p/inferno-os/source/browse/utils/6l/asm.c
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

#include	"l.h"

#define	Dbufslop	100

#define PADDR(a)	((ulong)(a) & ~0x80000000)

vlong
entryvalue(void)
{
	char *a;
	Sym *s;

	a = INITENTRY;
	if(*a >= '0' && *a <= '9')
		return atolwhex(a);
	s = lookup(a, 0);
	if(s->type == 0)
		return INITTEXT;
	switch(s->type) {
	case STEXT:
		break;
	case SDATA:
		if(dlm)
			return s->value+INITDAT;
	default:
		diag("entry not text: %s", s->name);
	}
	return s->value;
}

void
wputl(ushort w)
{
	cput(w);
	cput(w>>8);
}

void
wput(ushort w)
{
	cput(w>>8);
	cput(w);
}

void
lput(long l)
{
	cput(l>>24);
	cput(l>>16);
	cput(l>>8);
	cput(l);
}

void
llput(vlong v)
{
	lput(v>>32);
	lput(v);
}

void
lputl(long l)
{
	cput(l);
	cput(l>>8);
	cput(l>>16);
	cput(l>>24);
}

void
llputl(vlong v)
{
	lputl(v);
	lputl(v>>32);
}

void
strnput(char *s, int n)
{
	int i;

	for(i=0; i<n; i++) {
		cput(*s);
		if(*s != 0)
			s++;
	}
}

void
asmb(void)
{
	Prog *p;
	long v, magic;
	int a;
	uchar *op1;
	vlong vl, va, fo, w;

	if(debug['v'])
		Bprint(&bso, "%5.2f asmb\n", cputime());
	Bflush(&bso);

	seek(cout, HEADR, 0);
	pc = INITTEXT;
	curp = firstp;
	for(p = firstp; p != P; p = p->link) {
		if(p->as == ATEXT)
			curtext = p;
		if(p->pc != pc) {
			if(!debug['a'])
				print("%P\n", curp);
			diag("phase error %llux sb %llux in %s", p->pc, pc, TNAME);
			pc = p->pc;
		}
		curp = p;
		asmins(p);
		a = (andptr - and);
		if(cbc < a)
			cflush();
		if(debug['a']) {
			Bprint(&bso, pcstr, pc);
			for(op1 = and; op1 < andptr; op1++)
				Bprint(&bso, "%.2ux", *op1);
			for(; op1 < and+Maxand; op1++)
				Bprint(&bso, "  ");
			Bprint(&bso, "%P\n", curp);
		}
		if(dlm) {
			if(p->as == ATEXT)
				reloca = nil;
			else if(reloca != nil)
				diag("reloc failure: %P", curp);
		}
		memmove(cbp, and, a);
		cbp += a;
		pc += a;
		cbc -= a;
	}
	cflush();


	switch(HEADTYPE) {
	default:
		diag("unknown header type %ld", HEADTYPE);
	case 2:
	case 5:
		seek(cout, HEADR+textsize, 0);
		break;
	case 6:
		v = HEADR+textsize;
		myseek(cout, v);
		v = rnd(v, 4096) - v;
		while(v > 0) {
			cput(0);
			v--;
		}
		cflush();
		break;

	case 7:
		v = rnd(HEADR+textsize, INITRND);
		myseek(cout, v);
		break;
	}

	if(debug['v'])
		Bprint(&bso, "%5.2f datblk\n", cputime());
	Bflush(&bso);

	if(dlm){
		char buf[8];

		write(cout, buf, INITDAT-textsize);
		textsize = INITDAT;
	}

	for(v = 0; v < datsize; v += sizeof(buf)-Dbufslop) {
		if(datsize-v > sizeof(buf)-Dbufslop)
			datblk(v, sizeof(buf)-Dbufslop);
		else
			datblk(v, datsize-v);
	}

	symsize = 0;
	spsize = 0;
	lcsize = 0;
	if(!debug['s']) {
		if(debug['v'])
			Bprint(&bso, "%5.2f sym\n", cputime());
		Bflush(&bso);
		switch(HEADTYPE) {
		default:
		case 2:
		case 5:
debug['s'] = 1;
			seek(cout, HEADR+textsize+datsize, 0);
			break;
		case 7:
debug['s'] = 1;
			seek(cout, HEADR+textsize+datsize, 0);
			linuxstrtable();
			break;
		case 6:
			debug['s'] = 1;
			break;
		}
		if(!debug['s'])
			asmsym();
		if(debug['v'])
			Bprint(&bso, "%5.2f sp\n", cputime());
		Bflush(&bso);
		if(debug['v'])
			Bprint(&bso, "%5.2f pc\n", cputime());
		Bflush(&bso);
		if(!debug['s'])
			asmlc();
		if(dlm)
			asmdyn();
		cflush();
	} else
	if(dlm){
		seek(cout, HEADR+textsize+datsize, 0);
		asmdyn();
		cflush();
	}

	if(debug['v'])
		Bprint(&bso, "%5.2f headr\n", cputime());
	Bflush(&bso);
	seek(cout, 0L, 0);
	switch(HEADTYPE) {
	default:
	case 2:	/* plan9 */
		magic = 4*26*26+7;
		magic |= 0x00008000;		/* fat header */
		if(dlm)
			magic |= 0x80000000;	/* dlm */
		lput(magic);			/* magic */
		lput(textsize);			/* sizes */
		lput(datsize);
		lput(bsssize);
		lput(symsize);			/* nsyms */
		vl = entryvalue();
		lput(PADDR(vl));		/* va of entry */
		lput(spsize);			/* sp offsets */
		lput(lcsize);			/* line offsets */
		llput(vl);			/* va of entry */
		break;
	case 3:	/* plan9 */
		magic = 4*26*26+7;
		if(dlm)
			magic |= 0x80000000;
		lput(magic);			/* magic */
		lput(textsize);			/* sizes */
		lput(datsize);
		lput(bsssize);
		lput(symsize);			/* nsyms */
		lput(entryvalue());		/* va of entry */
		lput(spsize);			/* sp offsets */
		lput(lcsize);			/* line offsets */
		break;
	case 5:
		strnput("\177ELF", 4);		/* e_ident */
		cput(1);			/* class = 32 bit */
		cput(1);			/* data = LSB */
		cput(1);			/* version = CURRENT */
		strnput("", 9);
		wputl(2);			/* type = EXEC */
		wputl(62);			/* machine = AMD64 */
		lputl(1L);			/* version = CURRENT */
		lputl(PADDR(entryvalue()));	/* entry vaddr */
		lputl(52L);			/* offset to first phdr */
		lputl(0L);			/* offset to first shdr */
		lputl(0L);			/* processor specific flags */
		wputl(52);			/* Ehdr size */
		wputl(32);			/* Phdr size */
		wputl(3);			/* # of Phdrs */
		wputl(40);			/* Shdr size */
		wputl(0);			/* # of Shdrs */
		wputl(0);			/* Shdr string size */

		lputl(1L);			/* text - type = PT_LOAD */
		lputl(HEADR);			/* file offset */
		lputl(INITTEXT);		/* vaddr */
		lputl(PADDR(INITTEXT));		/* paddr */
		lputl(textsize);		/* file size */
		lputl(textsize);		/* memory size */
		lputl(0x05L);			/* protections = RX */
		lputl(INITRND);			/* alignment */

		lputl(1L);			/* data - type = PT_LOAD */
		lputl(HEADR+textsize);		/* file offset */
		lputl(INITDAT);			/* vaddr */
		lputl(PADDR(INITDAT));		/* paddr */
		lputl(datsize);			/* file size */
		lputl(datsize+bsssize);		/* memory size */
		lputl(0x06L);			/* protections = RW */
		lputl(INITRND);			/* alignment */

		lputl(0L);			/* data - type = PT_NULL */
		lputl(HEADR+textsize+datsize);	/* file offset */
		lputl(0L);
		lputl(0L);
		lputl(symsize);			/* symbol table size */
		lputl(lcsize);			/* line number size */
		lputl(0x04L);			/* protections = R */
		lputl(0x04L);			/* alignment */
		break;
	case 6:
		/* apple MACH */
		va = 4096;

		lputl(0xfeedfacf);		/* 64-bit */
		lputl((1<<24)|7);		/* cputype - x86/ABI64 */
		lputl(3);			/* subtype - x86 */
		lputl(2);			/* file type - mach executable */
		lputl(4);			/* number of loads */
		lputl(machheadr()-32);		/* size of loads */
		lputl(1);			/* flags - no undefines */
		lputl(0);			/* reserved */

		machseg("__PAGEZERO",
			0,va,			/* vaddr vsize */
			0,0,			/* fileoffset filesize */
			0,0,			/* protects */
			0,0);			/* sections flags */

		v = rnd(HEADR+textsize, INITRND);
		machseg("__TEXT",
			va,			/* vaddr */
			v,			/* vsize */
			0,v,			/* fileoffset filesize */
			7,5,			/* protects */
			1,0);			/* sections flags */
		machsect("__text", "__TEXT",
			va+HEADR,v-HEADR,	/* addr size */
			HEADR,0,0,0,		/* offset align reloc nreloc */
			0|0x400);		/* flag - some instructions */

		w = datsize+bsssize;
		machseg("__DATA",
			va+v,			/* vaddr */
			w,			/* vsize */
			v,datsize,		/* fileoffset filesize */
			7,3,			/* protects */
			2,0);			/* sections flags */
		machsect("__data", "__DATA",
			va+v,datsize,		/* addr size */
			v,0,0,0,		/* offset align reloc nreloc */
			0);			/* flag */
		machsect("__bss", "__DATA",
			va+v+datsize,bsssize,	/* addr size */
			0,0,0,0,		/* offset align reloc nreloc */
			1);			/* flag - zero fill */
		machstack(va+HEADR);
		break;
	case 7:
		/* elf amd-64 */
		strnput("\177ELF", 4);		/* e_ident */
		cput(2);			/* class = 64 bit */
		cput(1);			/* data = LSB */
		cput(1);			/* version = CURRENT */
		strnput("", 9);

		wputl(2);			/* type = EXEC */
		wputl(62);			/* machine = AMD64 */
		lputl(1L);			/* version = CURRENT */
		llputl(entryvalue());		/* entry vaddr */
		llputl(64L);			/* offset to first phdr */
		llputl(64L+56*3);		/* offset to first shdr */
		lputl(0L);			/* processor specific flags */
		wputl(64);			/* Ehdr size */
		wputl(56);			/* Phdr size */
		wputl(3);			/* # of Phdrs */
		wputl(64);			/* Shdr size */
		wputl(5);			/* # of Shdrs */
		wputl(4);			/* Shdr with strings */

fo = 0;
va = INITRND;
w = HEADR+textsize;


		linuxphdr(1,			/* text - type = PT_LOAD */
			1L+4L,			/* text - flags = PF_X+PF_R */
			0,			/* file offset */
			va,			/* vaddr */
			va,			/* paddr */
			w,			/* file size */
			w,			/* memory size */
			INITRND);		/* alignment */

fo = rnd(fo+w, INITRND);
va = rnd(va+w, INITRND);
w = datsize;

		linuxphdr(1,			/* data - type = PT_LOAD */
			2L+4L,			/* data - flags = PF_W+PF_R */
			fo,			/* file offset */
			va,			/* vaddr */
			va,			/* paddr */
			w,			/* file size */
			w+bsssize,		/* memory size */
			INITRND);		/* alignment */

		linuxphdr(0x6474e551,		/* gok - type = gok */
			1L+2L+4L,		/* gok - flags = PF_X+PF_R */
			0,			/* file offset */
			0,			/* vaddr */
			0,			/* paddr */
			0,			/* file size */
			0,			/* memory size */
			8);			/* alignment */

		linuxshdr(nil,			/* name */
			0,			/* type */
			0,			/* flags */
			0,			/* addr */
			0,			/* off */
			0,			/* size */
			0,			/* link */
			0,			/* info */
			0,			/* align */
			0);			/* entsize */

stroffset = 1;
fo = 0;
va = INITRND;
w = HEADR+textsize;

		linuxshdr(".text",		/* name */
			1,			/* type */
			6,			/* flags */
			va,			/* addr */
			fo,			/* off */
			w,			/* size */
			0,			/* link */
			0,			/* info */
			8,			/* align */
			0);			/* entsize */

fo = rnd(fo+w, INITRND);
va = rnd(va+w, INITRND);
w = datsize;

		linuxshdr(".data",		/* name */
			1,			/* type */
			3,			/* flags */
			va,			/* addr */
			fo,			/* off */
			w,			/* size */
			0,			/* link */
			0,			/* info */
			8,			/* align */
			0);			/* entsize */

fo += w;
va += w;
w = bsssize;

		linuxshdr(".bss",		/* name */
			8,			/* type */
			3,			/* flags */
			va,			/* addr */
			fo,			/* off */
			w,			/* size */
			0,			/* link */
			0,			/* info */
			8,			/* align */
			0);			/* entsize */

fo = HEADR+textsize+datsize;
w = stroffset +
	strlen(".shstrtab")+1;
//	strlen(".gosymtab")+1;

		linuxshdr(".shstrtab",		/* name */
			3,			/* type */
			0,			/* flags */
			0,			/* addr */
			fo,			/* off */
			w,			/* size */
			0,			/* link */
			0,			/* info */
			8,			/* align */
			0);			/* entsize */

//fo += w;
//
//		linuxshdr(".gosymtab",		/* name */
//			2,			/* type */
//			0,			/* flags */
//			0,			/* addr */
//			fo,			/* off */
//			0,			/* size */
//			0,			/* link */
//			0,			/* info */
//			8,			/* align */
//			0);			/* entsize */
		break;
	}
	cflush();
}

void
cflush(void)
{
	int n;

	n = sizeof(buf.cbuf) - cbc;
	if(n)
		write(cout, buf.cbuf, n);
	cbp = buf.cbuf;
	cbc = sizeof(buf.cbuf);
}

void
outa(int n, uchar *cast, uchar *map, vlong l)
{
	int i, j;

	Bprint(&bso, pcstr, l);
	for(i=0; i<n; i++) {
		j = i;
		if(map != nil)
			j = map[j];
		Bprint(&bso, "%.2ux", cast[j]);
	}
	for(; i<Maxand; i++)
		Bprint(&bso, "  ");
	Bprint(&bso, "%P\n", curp);
}

void
datblk(long s, long n)
{
	Prog *p;
	uchar *cast;
	long l, fl, j;
	vlong o;
	int i, c;

	memset(buf.dbuf, 0, n+Dbufslop);
	for(p = datap; p != P; p = p->link) {
		curp = p;
		l = p->from.sym->value + p->from.offset - s;
		c = p->from.scale;
		i = 0;
		if(l < 0) {
			if(l+c <= 0)
				continue;
			while(l < 0) {
				l++;
				i++;
			}
		}
		if(l >= n)
			continue;
		if(p->as != AINIT && p->as != ADYNT) {
			for(j=l+(c-i)-1; j>=l; j--)
				if(buf.dbuf[j]) {
					print("%P\n", p);
					diag("multiple initialization");
					break;
				}
		}

		switch(p->to.type) {
		case D_FCONST:
			switch(c) {
			default:
			case 4:
				fl = ieeedtof(&p->to.ieee);
				cast = (uchar*)&fl;
				if(debug['a'] && i == 0)
					outa(c, cast, fnuxi4, l+s+INITDAT);
				for(; i<c; i++) {
					buf.dbuf[l] = cast[fnuxi4[i]];
					l++;
				}
				break;
			case 8:
				cast = (uchar*)&p->to.ieee;
				if(debug['a'] && i == 0)
					outa(c, cast, fnuxi8, l+s+INITDAT);
				for(; i<c; i++) {
					buf.dbuf[l] = cast[fnuxi8[i]];
					l++;
				}
				break;
			}
			break;

		case D_SCONST:
			if(debug['a'] && i == 0)
				outa(c, (uchar*)p->to.scon, nil, l+s+INITDAT);
			for(; i<c; i++) {
				buf.dbuf[l] = p->to.scon[i];
				l++;
			}
			break;
		default:
			o = p->to.offset;
			if(p->to.type == D_ADDR) {
				if(p->to.index != D_STATIC && p->to.index != D_EXTERN)
					diag("DADDR type%P", p);
				if(p->to.sym) {
					if(p->to.sym->type == SUNDEF)
						ckoff(p->to.sym, o);
					o += p->to.sym->value;
					if(p->to.sym->type != STEXT && p->to.sym->type != SUNDEF)
						o += INITDAT;
					if(dlm)
						dynreloc(p->to.sym, l+s+INITDAT, 1);
				}
			}
			fl = o;
			cast = (uchar*)&fl;
			switch(c) {
			default:
				diag("bad nuxi %d %d\n%P", c, i, curp);
				break;
			case 1:
				if(debug['a'] && i == 0)
					outa(c, cast, inuxi1, l+s+INITDAT);
				for(; i<c; i++) {
					buf.dbuf[l] = cast[inuxi1[i]];
					l++;
				}
				break;
			case 2:
				if(debug['a'] && i == 0)
					outa(c, cast, inuxi2, l+s+INITDAT);
				for(; i<c; i++) {
					buf.dbuf[l] = cast[inuxi2[i]];
					l++;
				}
				break;
			case 4:
				if(debug['a'] && i == 0)
					outa(c, cast, inuxi4, l+s+INITDAT);
				for(; i<c; i++) {
					buf.dbuf[l] = cast[inuxi4[i]];
					l++;
				}
				break;
			case 8:
				cast = (uchar*)&o;
				if(debug['a'] && i == 0)
					outa(c, cast, inuxi8, l+s+INITDAT);
				for(; i<c; i++) {
					buf.dbuf[l] = cast[inuxi8[i]];
					l++;
				}
				break;
			}
			break;
		}
	}
	write(cout, buf.dbuf, n);
}

vlong
rnd(vlong v, vlong r)
{
	vlong c;

	if(r <= 0)
		return v;
	v += r - 1;
	c = v % r;
	if(c < 0)
		c += r;
	v -= c;
	return v;
}

void
vputl(vlong v)
{
	lputl(v);
	lputl(v>>32);
}

void
machseg(char *name, vlong vaddr, vlong vsize, vlong foff, vlong fsize,
	ulong prot1, ulong prot2, ulong nsect, ulong flag)
{
	lputl(25);	// section
	lputl(72 + 80*nsect);
	strnput(name, 16);
	vputl(vaddr);
	vputl(vsize);
	vputl(foff);
	vputl(fsize);
	lputl(prot1);
	lputl(prot2);
	lputl(nsect);
	lputl(flag);
}

void
machsect(char *name, char *seg, vlong addr, vlong size, ulong off,
	ulong align, ulong reloc, ulong nreloc, ulong flag)
{
	strnput(name, 16);
	strnput(seg, 16);
	vputl(addr);
	vputl(size);
	lputl(off);
	lputl(align);
	lputl(reloc);
	lputl(nreloc);
	lputl(flag);
	lputl(0);	/* reserved */
	lputl(0);	/* reserved */
	lputl(0);	/* reserved */
}

void
machstack(vlong e)
{
	int i;

	lputl(5);			/* unix thread */
	lputl((42+4)*4);		/* total byte count */

	lputl(4);			/* thread type */
	lputl(42);			/* word count */

	for(i=0; i<32; i++)
		lputl(0);
	vputl(e);
	for(i=0; i<8; i++)
		lputl(0);
}

ulong
machheadr(void)
{
	ulong a;

	a = 8;		/* a.out header */
	a += 18;	/* page zero seg */
	a += 18;	/* text seg */
	a += 20;	/* text sect */
	a += 18;	/* data seg */
	a += 20;	/* data sect */
	a += 20;	/* bss sect */
	a += 46;	/* stack sect */

	return a*4;
}

ulong
linuxheadr(void)
{
	ulong a;

	a = 64;		/* a.out header */

	a += 56;	/* page zero seg */
	a += 56;	/* text seg */
	a += 56;	/* stack seg */

	a += 64;	/* nil sect */
	a += 64;	/* .text sect */
	a += 64;	/* .data seg */
	a += 64;	/* .bss sect */
	a += 64;	/* .shstrtab sect - strings for headers */
//	a += 64;	/* .gosymtab sect */

	return a;
}


void
linuxphdr(int type, int flags, vlong foff,
	vlong vaddr, vlong paddr,
	vlong filesize, vlong memsize, vlong align)
{

	lputl(type);			/* text - type = PT_LOAD */
	lputl(flags);			/* text - flags = PF_X+PF_R */
	llputl(foff);			/* file offset */
	llputl(vaddr);			/* vaddr */
	llputl(paddr);			/* paddr */
	llputl(filesize);		/* file size */
	llputl(memsize);		/* memory size */
	llputl(align);			/* alignment */
}

void
linuxshdr(char *name, ulong type, vlong flags, vlong addr, vlong off,
	vlong size, ulong link, ulong info, vlong align, vlong entsize)
{
	lputl(stroffset);
	lputl(type);
	llputl(flags);
	llputl(addr);
	llputl(off);
	llputl(size);
	lputl(link);
	lputl(info);
	llputl(align);
	llputl(entsize);

	if(name != nil)
		stroffset += strlen(name)+1;
}

void
linuxstrtable(void)
{
	char *name;

	name = "";
	strnput(name, strlen(name)+1);
	name = ".text";
	strnput(name, strlen(name)+1);
	name = ".data";
	strnput(name, strlen(name)+1);
	name = ".bss";
	strnput(name, strlen(name)+1);
	name = ".shstrtab";
	strnput(name, strlen(name)+1);
//	name = ".gosymtab";
//	strnput(name, strlen(name)+1);
}
