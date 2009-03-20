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

#include	"l.h"

#define	Dbufslop	100

uint32 symdatva = 0x99<<24;
uint32 stroffset;
uint32 strtabsize;

uint32 machheadr(void);
uint32		elfheadr(void);
void		elfphdr(int type, int flags, uint32 foff, uint32 vaddr, uint32 paddr, uint32 filesize, uint32 memsize, uint32 align);
void		elfshdr(char *name, uint32 type, uint32 flags, uint32 addr, uint32 off, uint32 size, uint32 link, uint32 info, uint32 align, uint32 entsize);
int		elfstrtable(void);
void		machdylink(void);
uint32		machheadr(void);
void		machsect(char *name, char *seg, vlong addr, vlong size, uint32 off, uint32 align, uint32 reloc, uint32 nreloc, uint32 flag);
void		machseg(char *name, uint32 vaddr, uint32 vsize, uint32 foff, uint32 fsize, uint32 prot1, uint32 prot2, uint32 nsect, uint32 flag);
void		machstack(vlong e);
void		machsymseg(uint32 foffset, uint32 fsize);

int32
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
lput(int32 l)
{
	cput(l>>24);
	cput(l>>16);
	cput(l>>8);
	cput(l);
}

void
lputl(int32 l)
{
	cput(l);
	cput(l>>8);
	cput(l>>16);
	cput(l>>24);
}

void
vputl(uvlong l)
{
	lputl(l >> 32);
	lputl(l);
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

void
asmb(void)
{
	Prog *p;
	int32 v, magic;
	int a, np, nl, ns;
	uint32 va, fo, w, symo;
	uchar *op1;

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
			diag("phase error %lux sb %lux in %s", p->pc, pc, TNAME);
			pc = p->pc;
		}
		curp = p;
		asmins(p);
		if(cbc < sizeof(and))
			cflush();
		a = (andptr - and);
		if(debug['a']) {
			Bprint(&bso, pcstr, pc);
			for(op1 = and; op1 < andptr; op1++)
				Bprint(&bso, "%.2ux", *op1 & 0xff);
			Bprint(&bso, "\t%P\n", curp);
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
		diag("unknown header type %d", HEADTYPE);
	case 0:
		seek(cout, rnd(HEADR+textsize, 8192), 0);
		break;
	case 1:
		textsize = rnd(HEADR+textsize, 4096)-HEADR;
		seek(cout, textsize+HEADR, 0);
		break;
	case 2:
		seek(cout, HEADR+textsize, 0);
		break;
	case 3:
	case 4:
		seek(cout, HEADR+rnd(textsize, INITRND), 0);
		break;
	case 6:
		v = HEADR+textsize;
		seek(cout, v, 0);
		v = rnd(v, 4096) - v;
		while(v > 0) {
			cput(0);
			v--;
		}
		cflush();
		break;
	case 7:
		seek(cout, rnd(HEADR+textsize, INITRND)+datsize, 0);
		strtabsize = elfstrtable();
		cflush();
		v = rnd(HEADR+textsize, INITRND);
		seek(cout, v, 0);
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
	symo = 0;
	if(!debug['s']) {
		if(debug['v'])
			Bprint(&bso, "%5.2f sym\n", cputime());
		Bflush(&bso);
		switch(HEADTYPE) {
		default:
		case 0:
			seek(cout, rnd(HEADR+textsize, 8192)+datsize, 0);
			break;
		case 1:
			seek(cout, rnd(HEADR+textsize, INITRND)+datsize, 0);
			break;
		case 2:
			seek(cout, HEADR+textsize+datsize, 0);
			break;
		case 3:
		case 4:
			debug['s'] = 1;
			symo = HEADR+textsize+datsize;
			break;
		case 6:
			symo = rnd(HEADR+textsize, INITRND)+rnd(datsize, INITRND);
			break;
		case 7:
			symo = rnd(HEADR+textsize, INITRND)+datsize+strtabsize;
			symo = rnd(symo, INITRND);
			break;
		}
		seek(cout, symo+8, 0);
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
		seek(cout, symo, 0);
		lputl(symsize);
		lputl(lcsize);
		cflush();
	}
	else if(dlm){
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
	case 0:	/* garbage */
		lput(0x160L<<16);		/* magic and sections */
		lput(0L);			/* time and date */
		lput(rnd(HEADR+textsize, 4096)+datsize);
		lput(symsize);			/* nsyms */
		lput((0x38L<<16)|7L);		/* size of optional hdr and flags */
		lput((0413<<16)|0437L);		/* magic and version */
		lput(rnd(HEADR+textsize, 4096));	/* sizes */
		lput(datsize);
		lput(bsssize);
		lput(entryvalue());		/* va of entry */
		lput(INITTEXT-HEADR);		/* va of base of text */
		lput(INITDAT);			/* va of base of data */
		lput(INITDAT+datsize);		/* va of base of bss */
		lput(~0L);			/* gp reg mask */
		lput(0L);
		lput(0L);
		lput(0L);
		lput(0L);
		lput(~0L);			/* gp value ?? */
		break;
		lputl(0);			/* x */
	case 1:	/* unix coff */
		/*
		 * file header
		 */
		lputl(0x0004014c);		/* 4 sections, magic */
		lputl(0);			/* unix time stamp */
		lputl(0);			/* symbol table */
		lputl(0);			/* nsyms */
		lputl(0x0003001c);		/* flags, sizeof a.out header */
		/*
		 * a.out header
		 */
		lputl(0x10b);			/* magic, version stamp */
		lputl(rnd(textsize, INITRND));	/* text sizes */
		lputl(datsize);			/* data sizes */
		lputl(bsssize);			/* bss sizes */
		lput(entryvalue());		/* va of entry */
		lputl(INITTEXT);		/* text start */
		lputl(INITDAT);			/* data start */
		/*
		 * text section header
		 */
		s8put(".text");
		lputl(HEADR);			/* pa */
		lputl(HEADR);			/* va */
		lputl(textsize);		/* text size */
		lputl(HEADR);			/* file offset */
		lputl(0);			/* relocation */
		lputl(0);			/* line numbers */
		lputl(0);			/* relocation, line numbers */
		lputl(0x20);			/* flags text only */
		/*
		 * data section header
		 */
		s8put(".data");
		lputl(INITDAT);			/* pa */
		lputl(INITDAT);			/* va */
		lputl(datsize);			/* data size */
		lputl(HEADR+textsize);		/* file offset */
		lputl(0);			/* relocation */
		lputl(0);			/* line numbers */
		lputl(0);			/* relocation, line numbers */
		lputl(0x40);			/* flags data only */
		/*
		 * bss section header
		 */
		s8put(".bss");
		lputl(INITDAT+datsize);		/* pa */
		lputl(INITDAT+datsize);		/* va */
		lputl(bsssize);			/* bss size */
		lputl(0);			/* file offset */
		lputl(0);			/* relocation */
		lputl(0);			/* line numbers */
		lputl(0);			/* relocation, line numbers */
		lputl(0x80);			/* flags bss only */
		/*
		 * comment section header
		 */
		s8put(".comment");
		lputl(0);			/* pa */
		lputl(0);			/* va */
		lputl(symsize+lcsize);		/* comment size */
		lputl(HEADR+textsize+datsize);	/* file offset */
		lputl(HEADR+textsize+datsize);	/* offset of syms */
		lputl(HEADR+textsize+datsize+symsize);/* offset of line numbers */
		lputl(0);			/* relocation, line numbers */
		lputl(0x200);			/* flags comment only */
		break;
	case 2:	/* plan9 */
		magic = 4*11*11+7;
		if(dlm)
			magic |= 0x80000000;
		lput(magic);		/* magic */
		lput(textsize);			/* sizes */
		lput(datsize);
		lput(bsssize);
		lput(symsize);			/* nsyms */
		lput(entryvalue());		/* va of entry */
		lput(spsize);			/* sp offsets */
		lput(lcsize);			/* line offsets */
		break;
	case 3:
		/* MS-DOS .COM */
		break;
	case 4:
		/* fake MS-DOS .EXE */
		v = rnd(HEADR+textsize, INITRND)+datsize;
		wputl(0x5A4D);			/* 'MZ' */
		wputl(v % 512);			/* bytes in last page */
		wputl(rnd(v, 512)/512);		/* total number of pages */
		wputl(0x0000);			/* number of reloc items */
		v = rnd(HEADR-(INITTEXT & 0xFFFF), 16);
		wputl(v/16);			/* size of header */
		wputl(0x0000);			/* minimum allocation */
		wputl(0xFFFF);			/* maximum allocation */
		wputl(0x0000);			/* initial ss value */
		wputl(0x0100);			/* initial sp value */
		wputl(0x0000);			/* complemented checksum */
		v = entryvalue();
		wputl(v);			/* initial ip value (!) */
		wputl(0x0000);			/* initial cs value */
		wputl(0x0000);
		wputl(0x0000);
		wputl(0x003E);			/* reloc table offset */
		wputl(0x0000);			/* overlay number */
		break;
	
	case 6:
		/* apple MACH */
		va = 4096;

		lputl(0xfeedface);		/* 32-bit */
		lputl(7);		/* cputype - x86 */
		lputl(3);			/* subtype - x86 */
		lputl(2);			/* file type - mach executable */
		nl = 4;
		if (!debug['s'])
			nl += 3;
		if (!debug['d'])	// -d = turn off "dynamic loader"
			nl += 3;
		lputl(nl);			/* number of loads */
		lputl(machheadr()-28);		/* size of loads */
		lputl(1);			/* flags - no undefines */

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

		machdylink();
		machstack(entryvalue());

		if (!debug['s']) {
			machseg("__SYMDAT",
				symdatva,		/* vaddr */
				8+symsize+lcsize,		/* vsize */
				symo, 8+symsize+lcsize,	/* fileoffset filesize */
				7, 5,			/* protects */
				0, 0);			/* sections flags */

			machsymseg(symo+8,symsize);	/* fileoffset,filesize */
			machsymseg(symo+8+symsize,lcsize);	/* fileoffset,filesize */
		}
		break;

	case 7:
		np = 3;
		ns = 5;
		if(!debug['s']) {
			np++;
			ns += 2;
		}

		/* ELF header */
		strnput("\177ELF", 4);		/* e_ident */
		cput(1);			/* class = 32 bit */
		cput(1);			/* data = LSB */
		cput(1);			/* version = CURRENT */
		strnput("", 9);
		wputl(2);			/* type = EXEC */
		wputl(3);			/* machine = AMD64 */
		lputl(1L);			/* version = CURRENT */
		lputl(entryvalue());		/* entry vaddr */
		lputl(52L);			/* offset to first phdr */
		lputl(52L+32L*np);		/* offset to first shdr */
		lputl(0L);			/* processor specific flags */
		wputl(52L);			/* Ehdr size */
		wputl(32L);			/* Phdr size */
		wputl(np);			/* # of Phdrs */
		wputl(40L);			/* Shdr size */
		wputl(ns);			/* # of Shdrs */
		wputl(4);			/* Shdr with strings */

		/* prog headers */
		fo = 0;
		va = INITTEXT & ~((vlong)INITRND - 1);
		w = HEADR+textsize;

		elfphdr(1,			/* text - type = PT_LOAD */
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

		elfphdr(1,			/* data - type = PT_LOAD */
			2L+4L,			/* data - flags = PF_W+PF_R */
			fo,			/* file offset */
			va,			/* vaddr */
			va,			/* paddr */
			w,			/* file size */
			w+bsssize,		/* memory size */
			INITRND);		/* alignment */

		if(!debug['s']) {
			elfphdr(1,			/* data - type = PT_LOAD */
				2L+4L,			/* data - flags = PF_W+PF_R */
				symo,		/* file offset */
				symdatva,			/* vaddr */
				symdatva,			/* paddr */
				8+symsize+lcsize,			/* file size */
				8+symsize+lcsize,		/* memory size */
				INITRND);		/* alignment */
		}

		elfphdr(0x6474e551,		/* gok - type = gok */
			1L+2L+4L,		/* gok - flags = PF_X+PF_W+PF_R */
			0,			/* file offset */
			0,			/* vaddr */
			0,			/* paddr */
			0,			/* file size */
			0,			/* memory size */
			8);			/* alignment */

		/* segment headers */
		elfshdr(nil,			/* name */
			0,			/* type */
			0,			/* flags */
			0,			/* addr */
			0,			/* off */
			0,			/* size */
			0,			/* link */
			0,			/* info */
			0,			/* align */
			0);			/* entsize */

		stroffset = 1;  /* 0 means no name, so start at 1 */
		fo = HEADR;
		va = (INITTEXT & ~((vlong)INITRND - 1)) + HEADR;
		w = textsize;

		elfshdr(".text",		/* name */
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

		elfshdr(".data",		/* name */
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

		elfshdr(".bss",		/* name */
			8,			/* type */
			3,			/* flags */
			va,			/* addr */
			fo,			/* off */
			w,			/* size */
			0,			/* link */
			0,			/* info */
			8,			/* align */
			0);			/* entsize */

		w = strtabsize;

		elfshdr(".shstrtab",		/* name */
			3,			/* type */
			0,			/* flags */
			0,			/* addr */
			fo,			/* off */
			w,			/* size */
			0,			/* link */
			0,			/* info */
			1,			/* align */
			0);			/* entsize */

		if (debug['s'])
			break;

		fo = symo+8;
		w = symsize;

		elfshdr(".gosymtab",		/* name */
			1,			/* type 1 = SHT_PROGBITS */
			0,			/* flags */
			0,			/* addr */
			fo,			/* off */
			w,			/* size */
			0,			/* link */
			0,			/* info */
			1,			/* align */
			24);			/* entsize */

		fo += w;
		w = lcsize;

		elfshdr(".gopclntab",		/* name */
			1,			/* type 1 = SHT_PROGBITS*/
			0,			/* flags */
			0,			/* addr */
			fo,			/* off */
			w,			/* size */
			0,			/* link */
			0,			/* info */
			1,			/* align */
			24);			/* entsize */
		break;
	}
	cflush();
}

void
s8put(char *n)
{
	char name[8];
	int i;

	strncpy(name, n, sizeof(name));
	for(i=0; i<sizeof(name); i++)
		cput(name[i]);
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
datblk(int32 s, int32 n)
{
	Prog *p;
	char *cast;
	int32 l, fl, j;
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
				cast = (char*)&fl;
				if(debug['a'] && i == 0) {
					Bprint(&bso, pcstr, l+s+INITDAT);
					for(j=0; j<c; j++)
						Bprint(&bso, "%.2ux", cast[fnuxi4[j]] & 0xff);
					Bprint(&bso, "\t%P\n", curp);
				}
				for(; i<c; i++) {
					buf.dbuf[l] = cast[fnuxi4[i]];
					l++;
				}
				break;
			case 8:
				cast = (char*)&p->to.ieee;
				if(debug['a'] && i == 0) {
					Bprint(&bso, pcstr, l+s+INITDAT);
					for(j=0; j<c; j++)
						Bprint(&bso, "%.2ux", cast[fnuxi8[j]] & 0xff);
					Bprint(&bso, "\t%P\n", curp);
				}
				for(; i<c; i++) {
					buf.dbuf[l] = cast[fnuxi8[i]];
					l++;
				}
				break;
			}
			break;

		case D_SCONST:
			if(debug['a'] && i == 0) {
				Bprint(&bso, pcstr, l+s+INITDAT);
				for(j=0; j<c; j++)
					Bprint(&bso, "%.2ux", p->to.scon[j] & 0xff);
				Bprint(&bso, "\t%P\n", curp);
			}
			for(; i<c; i++) {
				buf.dbuf[l] = p->to.scon[i];
				l++;
			}
			break;
		default:
			fl = p->to.offset;
			if(p->to.type == D_ADDR) {
				if(p->to.index != D_STATIC && p->to.index != D_EXTERN)
					diag("DADDR type%P", p);
				if(p->to.sym) {
					if(p->to.sym->type == SUNDEF)
						ckoff(p->to.sym, fl);
					fl += p->to.sym->value;
					if(p->to.sym->type != STEXT && p->to.sym->type != SUNDEF)
						fl += INITDAT;
					if(dlm)
						dynreloc(p->to.sym, l+s+INITDAT, 1);
				}
			}
			cast = (char*)&fl;
			switch(c) {
			default:
				diag("bad nuxi %d %d\n%P", c, i, curp);
				break;
			case 1:
				if(debug['a'] && i == 0) {
					Bprint(&bso, pcstr, l+s+INITDAT);
					for(j=0; j<c; j++)
						Bprint(&bso, "%.2ux", cast[inuxi1[j]] & 0xff);
					Bprint(&bso, "\t%P\n", curp);
				}
				for(; i<c; i++) {
					buf.dbuf[l] = cast[inuxi1[i]];
					l++;
				}
				break;
			case 2:
				if(debug['a'] && i == 0) {
					Bprint(&bso, pcstr, l+s+INITDAT);
					for(j=0; j<c; j++)
						Bprint(&bso, "%.2ux", cast[inuxi2[j]] & 0xff);
					Bprint(&bso, "\t%P\n", curp);
				}
				for(; i<c; i++) {
					buf.dbuf[l] = cast[inuxi2[i]];
					l++;
				}
				break;
			case 4:
				if(debug['a'] && i == 0) {
					Bprint(&bso, pcstr, l+s+INITDAT);
					for(j=0; j<c; j++)
						Bprint(&bso, "%.2ux", cast[inuxi4[j]] & 0xff);
					Bprint(&bso, "\t%P\n", curp);
				}
				for(; i<c; i++) {
					buf.dbuf[l] = cast[inuxi4[i]];
					l++;
				}
				break;
			}
			break;
		}
	}
	write(cout, buf.dbuf, n);
}

int32
rnd(int32 v, int32 r)
{
	int32 c;

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
machseg(char *name, uint32 vaddr, uint32 vsize, uint32 foff, uint32 fsize,
	uint32 prot1, uint32 prot2, uint32 nsect, uint32 flag)
{
	lputl(1);	/* segment 32 */
	lputl(56 + 68*nsect);
	strnput(name, 16);
	lputl(vaddr);
	lputl(vsize);
	lputl(foff);
	lputl(fsize);
	lputl(prot1);
	lputl(prot2);
	lputl(nsect);
	lputl(flag);
}

void
machsymseg(uint32 foffset, uint32 fsize)
{
	lputl(3);	/* obsolete gdb debug info */
	lputl(16);	/* size of symseg command */
	lputl(foffset);
	lputl(fsize);
}

void
machsect(char *name, char *seg, vlong addr, vlong size, uint32 off,
	uint32 align, uint32 reloc, uint32 nreloc, uint32 flag)
{
	strnput(name, 16);
	strnput(seg, 16);
	lputl(addr);
	lputl(size);
	lputl(off);
	lputl(align);
	lputl(reloc);
	lputl(nreloc);
	lputl(flag);
	lputl(0);	/* reserved */
	lputl(0);	/* reserved */
}

// Emit a section requesting the dynamic loader
// but giving it no work to do (an empty dynamic symbol table).
// This is enough to make the Apple tracing programs (like dtrace)
// accept the binary, so that one can run dtruss on an 8.out.
void
machdylink(void)
{
	int i;

	if(debug['d'])
		return;

	lputl(2);	/* LC_SYMTAB */
	lputl(24);	/* byte count - 6 words*/
	for(i=0; i<4; i++)
		lputl(0);

	lputl(11);	/* LC_DYSYMTAB */
	lputl(80);	/* byte count - 20 words */
	for(i=0; i<18; i++)
		lputl(0);

	lputl(14);	/* LC_LOAD_DYLINKER */
	lputl(32);	/* byte count */
	lputl(12);	/* offset to string */
	strnput("/usr/lib/dyld", 32-12);
}

void
machstack(vlong e)
{
	int i;

	lputl(5);			/* unix thread */
	lputl((16+4)*4);		/* total byte count */

	lputl(1);			/* thread type - x86_THREAD_STATE32 */
	lputl(16);			/* word count */

	for(i=0; i<16; i++)	/* initial register set */
		if(i == 10)
			lputl(e);
		else
			lputl(0);
}

uint32
machheadr(void)
{
	uint32 a;
	enum {
		Header = 28,
		Seg = 56,
		Sect = 68,
		Symtab = 24,
		Dysymtab = 80,
		LoadDylinker = 32,
		Stack = 80,
		Symseg = 16,
	};

	a = Header;		/* a.out header */
	a += Seg;	/* page zero seg */
	a += Seg;	/* text seg */
	a += Sect;	/* text sect */
	a += Seg;	/* data seg */
	a += Sect;	/* data sect */
	a += Sect;	/* bss sect */
	if (!debug['d']) {
		a += Symtab;	/* symtab */
		a += Dysymtab;	/* dysymtab */
		a += LoadDylinker;	/* load dylinker */
	}
	a += Stack;	/* stack sect */
	if (!debug['s']) {
		a += Seg;	/* symdat seg */
		a += Symseg;	/* symtab seg */
		a += Symseg;	/* lctab seg */
	}

	return a;
}

uint32
elfheadr(void)
{
	uint32 a;

	a = 52;		/* elf header */

	a += 32;	/* page zero seg */
	a += 32;	/* text seg */
	a += 32;	/* stack seg */

	a += 40;	/* nil sect */
	a += 40;	/* .text sect */
	a += 40;	/* .data seg */
	a += 40;	/* .bss sect */
	a += 40;	/* .shstrtab sect - strings for headers */
	if (!debug['s']) {
		a += 32;	/* symdat seg */
		a += 40;	/* .gosymtab sect */
		a += 40;	/* .gopclntab sect */
	}

	return a;
}


void
elfphdr(int type, int flags, uint32 foff,
	uint32 vaddr, uint32 paddr,
	uint32 filesize, uint32 memsize, uint32 align)
{

	lputl(type);			/* text - type = PT_LOAD */
	lputl(foff);			/* file offset */
	lputl(vaddr);			/* vaddr */
	lputl(paddr);			/* paddr */
	lputl(filesize);		/* file size */
	lputl(memsize);		/* memory size */
	lputl(flags);			/* text - flags = PF_X+PF_R */
	lputl(align);			/* alignment */
}

void
elfshdr(char *name, uint32 type, uint32 flags, uint32 addr, uint32 off,
	uint32 size, uint32 link, uint32 info, uint32 align, uint32 entsize)
{
	lputl(stroffset);
	lputl(type);
	lputl(flags);
	lputl(addr);
	lputl(off);
	lputl(size);
	lputl(link);
	lputl(info);
	lputl(align);
	lputl(entsize);

	if(name != nil)
		stroffset += strlen(name)+1;
}

int
putstrtab(char* name)
{
	int w;

	w = strlen(name)+1;
	strnput(name, w);
	return w;
}

int
elfstrtable(void)
{
	int size;

	size = 0;
	size += putstrtab("");
	size += putstrtab(".text");
	size += putstrtab(".data");
	size += putstrtab(".bss");
	size += putstrtab(".shstrtab");
	if (!debug['s']) {
		size += putstrtab(".gosymtab");
		size += putstrtab(".gopclntab");
	}
	return size;
}
