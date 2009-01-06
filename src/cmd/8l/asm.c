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
wput(ushort w)
{
	cput(w);
	cput(w>>8);
}

void
wputb(ushort w)
{
	cput(w>>8);
	cput(w);
}

void
asmb(void)
{
	Prog *p;
	int32 v, magic;
	int a;
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
		wput(0x5A4D);			/* 'MZ' */
		wput(v % 512);			/* bytes in last page */
		wput(rnd(v, 512)/512);		/* total number of pages */
		wput(0x0000);			/* number of reloc items */
		v = rnd(HEADR-(INITTEXT & 0xFFFF), 16);
		wput(v/16);			/* size of header */
		wput(0x0000);			/* minimum allocation */
		wput(0xFFFF);			/* maximum allocation */
		wput(0x0000);			/* initial ss value */
		wput(0x0100);			/* initial sp value */
		wput(0x0000);			/* complemented checksum */
		v = entryvalue();
		wput(v);			/* initial ip value (!) */
		wput(0x0000);			/* initial cs value */
		wput(0x0000);
		wput(0x0000);
		wput(0x003E);			/* reloc table offset */
		wput(0x0000);			/* overlay number */
		break;
	}
	cflush();
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
