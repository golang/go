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
#include	"../ld/elf64.h"

#define	Dbufslop	100

#define PADDR(a)	((uint32)(a) & ~0x80000000)

char linuxdynld[] = "/lib64/ld-linux-x86-64.so.2";

char	zeroes[32];

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
wputl(uint16 w)
{
	cput(w);
	cput(w>>8);
}

void
wputb(uint16 w)
{
	cput(w>>8);
	cput(w);
}

void
lputb(int32 l)
{
	cput(l>>24);
	cput(l>>16);
	cput(l>>8);
	cput(l);
}

void
vputb(uint64 v)
{
	lputb(v>>32);
	lputb(v);
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
vputl(uint64 v)
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
	int32 v, magic;
	int a, nl;
	uchar *op1;
	vlong vl, va, startva, fo, w, symo, hashoff;
	vlong symdatva = 0x99LL<<32;
	Elf64Hdr *eh;
	Elf64PHdr *ph, *pph;
	Elf64SHdr *sh, *dynsh;

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
		debug['8'] = 1;	/* 64-bit addresses */
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
		debug['8'] = 1;	/* 64-bit addresses */
		v = rnd(HEADR+textsize, INITRND);
		seek(cout, v, 0);
		elf64init();
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
		case 2:
		case 5:
			debug['s'] = 1;
			symo = HEADR+textsize+datsize;
			break;
		case 6:
			symo = rnd(HEADR+textsize, INITRND)+rnd(datsize, INITRND);
			break;
		case 7:
			symo = rnd(HEADR+textsize, INITRND)+datsize;
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
		lputb(magic);			/* magic */
		lputb(textsize);			/* sizes */
		lputb(datsize);
		lputb(bsssize);
		lputb(symsize);			/* nsyms */
		vl = entryvalue();
		lputb(PADDR(vl));		/* va of entry */
		lputb(spsize);			/* sp offsets */
		lputb(lcsize);			/* line offsets */
		vputb(vl);			/* va of entry */
		break;
	case 3:	/* plan9 */
		magic = 4*26*26+7;
		if(dlm)
			magic |= 0x80000000;
		lputb(magic);			/* magic */
		lputb(textsize);		/* sizes */
		lputb(datsize);
		lputb(bsssize);
		lputb(symsize);			/* nsyms */
		lputb(entryvalue());		/* va of entry */
		lputb(spsize);			/* sp offsets */
		lputb(lcsize);			/* line offsets */
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
		nl = 4;
		if (!debug['s'])
			nl += 3;
		if (!debug['d'])	// -d = turn off "dynamic loader"
			nl += 3;
		lputl(nl);			/* number of loads */
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
		/* elf amd-64 */

		fo = 0;
		startva = INITTEXT - HEADR;
		va = startva;
		w = HEADR+textsize;

		/* This null SHdr must appear before all others */
		sh = newElf64SHdr("");

		pph = nil;	/* silence compiler */

		/* Dynamic linking sections */
		if (!debug['d']) {	/* -d suppresses dynamic loader format */

			/* P headers */
			/* program header info */
			pph = newElf64PHdr();
			pph->type = PT_PHDR;
			pph->flags = PF_R + PF_X;
			pph->off = ELF64HDRSIZE;
			pph->vaddr = startva + pph->off;
			pph->paddr = startva + pph->off;
			pph->align = 8;

			/* interpreter */
			ph = newElf64PHdr();
			ph->type = PT_INTERP;
			ph->flags = PF_R;
			ph->off = startelf();
			ph->vaddr = startva + ph->off;
			ph->paddr = startva + ph->off;
			write(cout, linuxdynld, sizeof linuxdynld);
			ph->filesz = endelf() - ph->off;
			ph->align = 1;

			sh = newElf64SHdr(".interp");
			sh->type = SHT_PROGBITS;
			sh->flags = SHF_ALLOC;
			sh->addr = va;
			sh->off = ph->off;
			sh->addr = startva + sh->off;
			sh->size = ph->filesz;
			sh->addralign = 1;

			/* dynamic load section */
			ph = newElf64PHdr();
			ph->type = PT_LOAD;
			ph->flags = PF_R + PF_W;
			ph->off = 0;
			ph->vaddr = startva + ph->off;
			ph->paddr = startva + ph->off;
			ph->align = 8;

			/* S headers inside dynamic load section */
			dynsh = newElf64SHdr(".dynamic");	// must be first
			dynsh->off = startelf();

			seek(cout, ELFDYNAMICSIZE, 1);	// leave room for dynamic table

			sh = newElf64SHdr(".hash");
			sh->type = SHT_HASH;
			sh->flags = SHF_ALLOC;
			sh->entsize = 4;
			sh->addr = va;
			sh->off = seek(cout, 0, 1);
			hashoff = sh->off;
			sh->addr = startva + sh->off;
			/* temporary hack: 8 zeroes means 0 buckets, 0 chains */
			write(cout, zeroes, 8);
			sh->size = endelf() - sh->off;
			sh->addralign = 8;

			sh = newElf64SHdr(".got");
			sh->type = SHT_PROGBITS;
			sh->flags = SHF_ALLOC+SHF_WRITE;
			sh->entsize = 8;
			sh->addr = va;
			sh->off = startelf();
			sh->addr = startva + sh->off;
			sh->size = endelf() - sh->off;
			sh->addralign = 8;

			sh = newElf64SHdr(".got.plt");
			sh->type = SHT_PROGBITS;
			sh->flags = SHF_ALLOC+SHF_WRITE;
			sh->entsize = 8;
			sh->addr = va;
			sh->off = startelf();
			sh->addr = startva + sh->off;
			sh->size = endelf() - sh->off;
			sh->addralign = 8;

			/* +8 necessary for now to silence readelf addressing at end of hash section */
			ph->filesz = endelf() - ph->off +8;	/* dynamic section maps these shdrs' data */
			ph->memsz = ph->filesz;

			dynsh->type = SHT_DYNAMIC;
			dynsh->flags = SHF_ALLOC+SHF_WRITE;
			dynsh->entsize = 16;
			dynsh->addr = startva + dynsh->off;
			seek(cout, dynsh->off, 0);
			elf64writedynent(DT_HASH, startva+hashoff);
			elf64writedynent(DT_STRTAB, startva+ELF64FULLHDRSIZE-STRTABSIZE);
			elf64writedynent(DT_SYMTAB, startva);
			elf64writedynent(DT_RELA, startva);
			elf64writedynent(DT_RELASZ, 0);	// size of the whole rela in bytes
			elf64writedynent(DT_RELAENT, ELF64RELASIZE);
			elf64writedynent(DT_STRSZ, STRTABSIZE);
			elf64writedynent(DT_SYMENT, 0);
			elf64writedynent(DT_NULL, 0);
			cflush();
			dynsh->size = seek(cout, 0, 1) - dynsh->off;
			dynsh->addralign = 8;

			/* dynamic section */
			ph = newElf64PHdr();
			ph->type = PT_DYNAMIC;
			ph->flags = PF_R + PF_W;
			ph->off = dynsh->off;
			ph->filesz = dynsh->size;
			ph->memsz = dynsh->size;
			ph->vaddr = startva + ph->off;
			ph->paddr = startva + ph->off;
			ph->align = 8;
		}

		ph = newElf64PHdr();
		ph->type = PT_LOAD;
		ph->flags = PF_X+PF_R;
		ph->vaddr = va + ELF64RESERVE;
		ph->paddr = va + ELF64RESERVE;
		ph->off = ELF64RESERVE;
		ph->filesz = w - ELF64RESERVE;
		ph->memsz = w - ELF64RESERVE;
		ph->align = INITRND;

		fo = rnd(fo+w, INITRND);
		va = rnd(va+w, INITRND);
		w = datsize;

		ph = newElf64PHdr();
		ph->type = PT_LOAD;
		ph->flags = PF_W+PF_R;
		ph->off = fo;
		ph->vaddr = va;
		ph->paddr = va;
		ph->filesz = w;
		ph->memsz = w+bsssize;
		ph->align = INITRND;

		if(!debug['s']) {
			ph = newElf64PHdr();
			ph->type = PT_LOAD;
			ph->flags = PF_W+PF_R;
			ph->off = symo;
			ph->vaddr = symdatva;
			ph->paddr = symdatva;
			ph->filesz = 8+symsize+lcsize;
			ph->memsz = 8+symsize+lcsize;
			ph->align = INITRND;
		}

		ph = newElf64PHdr();
		ph->type = 0x6474e551; 	/* GNU_STACK */
		ph->flags = PF_W+PF_R;
		ph->align = 8;

		fo = ELF64RESERVE;
		va = startva + fo;
		w = textsize;

		sh = newElf64SHdr(".text");
		sh->type = SHT_PROGBITS;
		sh->flags = SHF_ALLOC+SHF_EXECINSTR;
		sh->addr = va;
		sh->off = fo;
		sh->size = w;
		sh->addralign = 8;

		fo = rnd(fo+w, INITRND);
		va = rnd(va+w, INITRND);
		w = datsize;

		sh = newElf64SHdr(".data");
		sh->type = SHT_PROGBITS;
		sh->flags = SHF_WRITE+SHF_ALLOC;
		sh->addr = va;
		sh->off = fo;
		sh->size = w;
		sh->addralign = 8;

		fo += w;
		va += w;
		w = bsssize;

		sh = newElf64SHdr(".bss");
		sh->type = SHT_NOBITS;
		sh->flags = SHF_WRITE+SHF_ALLOC;
		sh->addr = va;
		sh->off = fo;
		sh->size = w;
		sh->addralign = 8;

		if (!debug['s']) {
			fo = symo+8;
			w = symsize;

			sh = newElf64SHdr(".gosymtab");
			sh->type = SHT_PROGBITS;
			sh->off = fo;
			sh->size = w;
			sh->addralign = 1;
			sh->entsize = 24;

			fo += w;
			w = lcsize;

			sh = newElf64SHdr(".gopclntab");
			sh->type = SHT_PROGBITS;
			sh->off = fo;
			sh->size = w;
			sh->addralign = 1;
			sh->entsize = 24;
		}

		sh = newElf64SHdr(".shstrtab");
		sh->type = SHT_STRTAB;
		sh->off = startelf();
		sh->addr = sh->off + startva;
		sh->addralign = 1;
		elf64writestrtable();
		sh->size = endelf() - sh->off;

		/* Main header */
		eh = getElf64Hdr();
		eh->ident[EI_MAG0] = '\177';
		eh->ident[EI_MAG1] = 'E';
		eh->ident[EI_MAG2] = 'L';
		eh->ident[EI_MAG3] = 'F';
		eh->ident[EI_CLASS] = ELFCLASS64;
		eh->ident[EI_DATA] = ELFDATA2LSB;
		eh->ident[EI_VERSION] = EV_CURRENT;

		eh->type = ET_EXEC;
		eh->machine = 62;	/* machine = AMD64 */
		eh->version = EV_CURRENT;
		eh->entry = entryvalue();

		if (!debug['d']) {
			pph->filesz = eh->phnum * ELF64PHDRSIZE;
			pph->memsz = pph->filesz;
		}

		seek(cout, 0, 0);
		a = 0;
		a += elf64writehdr();
		a += elf64writephdrs();
		a += elf64writeshdrs();
		if (a > ELF64FULLHDRSIZE) {
			diag("ELF64FULLHDRSIZE too small:", a);
		}
		cflush();

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
datblk(int32 s, int32 n)
{
	Prog *p;
	uchar *cast;
	int32 l, fl, j;
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
			i = -l;
			l = 0;
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

		case D_SBIG:
			if(debug['a'] && i == 0)
				outa(c, (uchar*)p->to.sbig, nil, l+s+INITDAT);
			for(; i<c; i++) {
				buf.dbuf[l] = p->to.sbig[i];
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
					if(p->to.sym->type == Sxxx) {
						curtext = p;	// show useful name in diag's output
						diag("missing symbol %s", p->to.sym->name);
					}
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
machseg(char *name, vlong vaddr, vlong vsize, vlong foff, vlong fsize,
	uint32 prot1, uint32 prot2, uint32 nsect, uint32 flag)
{
	lputl(25);	/* segment 64 */
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

// Emit a section requesting the dynamic loader
// but giving it no work to do (an empty dynamic symbol table).
// This is enough to make the Apple tracing programs (like dtrace)
// accept the binary, so that one can run dtruss on a 6.out.
// The dynamic linker loads at 0x8fe00000, so if we want to
// be able to build >2GB binaries, we're going to need to move
// the text segment to 4G like Apple does.
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
	lputl((42+4)*4);		/* total byte count */

	lputl(4);			/* thread type */
	lputl(42);			/* word count */

	for(i=0; i<32; i++)
		lputl(0);
	vputl(e);
	for(i=0; i<8; i++)
		lputl(0);
}

uint32
machheadr(void)
{
	uint32 a;

	a = 8;		/* a.out header */
	a += 18;	/* page zero seg */
	a += 18;	/* text seg */
	a += 20;	/* text sect */
	a += 18;	/* data seg */
	a += 20;	/* data sect */
	a += 20;	/* bss sect */
	a += 46;	/* stack sect */
	if (!debug['d']) {
		a += 6;	/* symtab */
		a += 20;	/* dysymtab */
		a += 8;	/* load dylinker */
	}
	if (!debug['s']) {
		a += 18;	/* symdat seg */
		a += 4;	/* symtab seg */
		a += 4;	/* lctab seg */
	}

	return a*4;
}
