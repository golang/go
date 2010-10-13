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

// Writing object files.

#include	"l.h"
#include	"../ld/lib.h"
#include	"../ld/elf.h"
#include	"../ld/dwarf.h"
#include	"../ld/macho.h"
#include	"../ld/pe.h"

#define	Dbufslop	100

char linuxdynld[] = "/lib/ld-linux.so.2";
char freebsddynld[] = "/usr/libexec/ld-elf.so.1";
uint32 symdatva = SYMDATVA;

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

vlong
addstring(Sym *s, char *str)
{
	int n, m;
	vlong r;
	Prog *p;

	if(s->type == 0)
		s->type = SDATA;
	s->reachable = 1;
	r = s->size;
	n = strlen(str)+1;
	if(strcmp(s->name, ".shstrtab") == 0)
		elfsetstring(str, r);
	while(n > 0) {
		m = n;
		if(m > sizeof(p->to.scon))
			m = sizeof(p->to.scon);
		p = newdata(s, s->size, m, D_EXTERN);
		p->to.type = D_SCONST;
		memmove(p->to.scon, str, m);
		s->size += m;
		str += m;
		n -= m;
	}
	return r;
}

vlong
adduintxx(Sym *s, uint64 v, int wid)
{
	vlong r;
	Prog *p;

	if(s->type == 0)
		s->type = SDATA;
	s->reachable = 1;
	r = s->size;
	p = newdata(s, s->size, wid, D_EXTERN);
	s->size += wid;
	p->to.type = D_CONST;
	p->to.offset = v;
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
	vlong r;
	Prog *p;
	enum { Ptrsize = 4 };

	if(s->type == 0)
		s->type = SDATA;
	s->reachable = 1;
	r = s->size;
	p = newdata(s, s->size, Ptrsize, D_EXTERN);
	s->size += Ptrsize;
	p->to.type = D_ADDR;
	p->to.index = D_EXTERN;
	p->to.offset = 0;
	p->to.sym = t;
	return r;
}

vlong
addsize(Sym *s, Sym *t)
{
	vlong r;
	Prog *p;
	enum { Ptrsize = 4 };

	if(s->type == 0)
		s->type = SDATA;
	s->reachable = 1;
	r = s->size;
	p = newdata(s, s->size, Ptrsize, D_EXTERN);
	s->size += Ptrsize;
	p->to.type = D_SIZE;
	p->to.index = D_EXTERN;
	p->to.offset = 0;
	p->to.sym = t;
	return r;
}

vlong
datoff(vlong addr)
{
	if(addr >= segdata.vaddr)
		return addr - segdata.vaddr + segdata.fileoff;
	diag("datoff %#llx", addr);
	return 0;
}

enum {
	ElfStrEmpty,
	ElfStrInterp,
	ElfStrHash,
	ElfStrGot,
	ElfStrGotPlt,
	ElfStrDynamic,
	ElfStrDynsym,
	ElfStrDynstr,
	ElfStrRel,
	ElfStrText,
	ElfStrData,
	ElfStrBss,
	ElfStrGosymcounts,
	ElfStrGosymtab,
	ElfStrGopclntab,
	ElfStrShstrtab,
	NElfStr
};

vlong elfstr[NElfStr];

static int
needlib(char *name)
{
	char *p;
	Sym *s;

	/* reuse hash code in symbol table */
	p = smprint(".dynlib.%s", name);
	s = lookup(p, 0);
	if(s->type == 0) {
		s->type = 100;	// avoid SDATA, etc.
		return 1;
	}
	return 0;
}

void
doelf(void)
{
	Sym *s, *shstrtab, *dynamic, *dynstr, *d;
	int h, nsym, t;

	if(!iself)
		return;

	/* predefine strings we need for section headers */
	shstrtab = lookup(".shstrtab", 0);
	shstrtab->reachable = 1;
	elfstr[ElfStrEmpty] = addstring(shstrtab, "");
	elfstr[ElfStrText] = addstring(shstrtab, ".text");
	elfstr[ElfStrData] = addstring(shstrtab, ".data");
	elfstr[ElfStrBss] = addstring(shstrtab, ".bss");
	addstring(shstrtab, ".elfdata");
	if(HEADTYPE == 8)
		addstring(shstrtab, ".closure");
	addstring(shstrtab, ".rodata");
	if(!debug['s']) {
		elfstr[ElfStrGosymcounts] = addstring(shstrtab, ".gosymcounts");
		elfstr[ElfStrGosymtab] = addstring(shstrtab, ".gosymtab");
		elfstr[ElfStrGopclntab] = addstring(shstrtab, ".gopclntab");
		dwarfaddshstrings(shstrtab);
	}
	elfstr[ElfStrShstrtab] = addstring(shstrtab, ".shstrtab");

	if(!debug['d']) {	/* -d suppresses dynamic loader format */
		elfstr[ElfStrInterp] = addstring(shstrtab, ".interp");
		elfstr[ElfStrHash] = addstring(shstrtab, ".hash");
		elfstr[ElfStrGot] = addstring(shstrtab, ".got");
		elfstr[ElfStrGotPlt] = addstring(shstrtab, ".got.plt");
		elfstr[ElfStrDynamic] = addstring(shstrtab, ".dynamic");
		elfstr[ElfStrDynsym] = addstring(shstrtab, ".dynsym");
		elfstr[ElfStrDynstr] = addstring(shstrtab, ".dynstr");
		elfstr[ElfStrRel] = addstring(shstrtab, ".rel");

		/* interpreter string */
		s = lookup(".interp", 0);
		s->reachable = 1;
		s->type = SELFDATA;

		/* dynamic symbol table - first entry all zeros */
		s = lookup(".dynsym", 0);
		s->type = SELFDATA;
		s->reachable = 1;
		s->size += ELF32SYMSIZE;

		/* dynamic string table */
		s = lookup(".dynstr", 0);
		s->reachable = 1;
		s->type = SELFDATA;
		addstring(s, "");
		dynstr = s;

		/* relocation table */
		s = lookup(".rel", 0);
		s->reachable = 1;
		s->type = SELFDATA;

		/* global offset table */
		s = lookup(".got", 0);
		s->reachable = 1;
		s->type = SELFDATA;

		/* got.plt - ??? */
		s = lookup(".got.plt", 0);
		s->reachable = 1;
		s->type = SELFDATA;

		/* define dynamic elf table */
		s = lookup(".dynamic", 0);
		s->reachable = 1;
		s->type = SELFDATA;
		dynamic = s;

		/*
		 * relocation entries for dynimport symbols
		 */
		nsym = 1;	// sym 0 is reserved
		for(h=0; h<NHASH; h++) {
			for(s=hash[h]; s!=S; s=s->hash) {
				if(!s->reachable || (s->type != STEXT && s->type != SDATA && s->type != SBSS) || s->dynimpname == nil)
					continue;

				if(!s->dynexport) {
					d = lookup(".rel", 0);
					addaddr(d, s);
					adduint32(d, ELF32_R_INFO(nsym, R_386_32));
				}

				nsym++;

				d = lookup(".dynsym", 0);
				adduint32(d, addstring(lookup(".dynstr", 0), s->dynimpname));
				/* value */
				if(!s->dynexport)
					adduint32(d, 0);
				else
					addaddr(d, s);

				/* size of object */
				adduint32(d, 0);

				/* type */
				t = STB_GLOBAL << 4;
				if(s->dynexport && s->type == STEXT)
					t |= STT_FUNC;
				else
					t |= STT_OBJECT;
				adduint8(d, t);

				/* reserved */
				adduint8(d, 0);

				/* section where symbol is defined */
				if(!s->dynexport)
					adduint16(d, SHN_UNDEF);
				else {
					switch(s->type) {
					default:
					case STEXT:
						t = 9;
						break;
					case SDATA:
						t = 10;
						break;
					case SBSS:
						t = 11;
						break;
					}
					adduint16(d, t);
				}

				if(!s->dynexport && needlib(s->dynimplib))
					elfwritedynent(dynamic, DT_NEEDED, addstring(dynstr, s->dynimplib));
			}
		}

		elfdynhash(nsym);

		/*
		 * .dynamic table
		 */
		s = dynamic;
		elfwritedynentsym(s, DT_HASH, lookup(".hash", 0));
		elfwritedynentsym(s, DT_SYMTAB, lookup(".dynsym", 0));
		elfwritedynent(s, DT_SYMENT, ELF32SYMSIZE);
		elfwritedynentsym(s, DT_STRTAB, lookup(".dynstr", 0));
		elfwritedynentsymsize(s, DT_STRSZ, lookup(".dynstr", 0));
		elfwritedynentsym(s, DT_REL, lookup(".rel", 0));
		elfwritedynentsymsize(s, DT_RELSZ, lookup(".rel", 0));
		elfwritedynent(s, DT_RELENT, ELF32RELSIZE);
		if(rpath)
			elfwritedynent(s, DT_RUNPATH, addstring(dynstr, rpath));
		elfwritedynent(s, DT_NULL, 0);
	}
}

void
shsym(Elf64_Shdr *sh, Sym *s)
{
	sh->addr = symaddr(s);
	sh->off = datoff(sh->addr);
	sh->size = s->size;
}

void
phsh(Elf64_Phdr *ph, Elf64_Shdr *sh)
{
	ph->vaddr = sh->addr;
	ph->paddr = ph->vaddr;
	ph->off = sh->off;
	ph->filesz = sh->size;
	ph->memsz = sh->size;
	ph->align = sh->addralign;
}

void
asmb(void)
{
	Prog *p;
	int32 v, magic;
	int a, dynsym;
	uint32 va, fo, w, symo, startva, machlink, erodata;
	uchar *op1;
	ulong expectpc;
	ElfEhdr *eh;
	ElfPhdr *ph, *pph;
	ElfShdr *sh;
	Section *sect;

	if(debug['v'])
		Bprint(&bso, "%5.2f asmb\n", cputime());
	Bflush(&bso);

	seek(cout, HEADR, 0);
	pc = INITTEXT;
	
	for(cursym = textp; cursym != nil; cursym = cursym->next) {
		for(p = cursym->text; p != P; p = p->link) {
			curp = p;
			if(HEADTYPE == 8) {
				// native client
				expectpc = p->pc;
				p->pc = pc;
				asmins(p);
				if(p->pc != expectpc) {
					Bflush(&bso);
					diag("phase error %ux sb %ux in %s", p->pc, expectpc, TNAME);
				}
				while(pc < p->pc) {
					cput(0x90);	// nop
					pc++;
				}
			}
			if(p->pc != pc) {
				Bflush(&bso);
				if(!debug['a'])
					print("%P\n", curp);
				diag("phase error %ux sb %ux in %s", p->pc, pc, TNAME);
				pc = p->pc;
			}
			if(HEADTYPE != 8) {
				asmins(p);
				if(pc != p->pc) {
					Bflush(&bso);
					diag("asmins changed pc %ux sb %ux in %s", p->pc, pc, TNAME);
				}
			}
			if(cbc < sizeof(and))
				cflush();
			a = (andptr - and);
	
			if(debug['a']) {
				Bprint(&bso, pcstr, pc);
				for(op1 = and; op1 < andptr; op1++)
					Bprint(&bso, "%.2ux", *op1 & 0xff);
				Bprint(&bso, "\t%P\n", curp);
			}
			memmove(cbp, and, a);
			cbp += a;
			pc += a;
			cbc -= a;
		}
	}
	if(HEADTYPE == 8) {
		int32 etext;
		
		etext = rnd(segtext.vaddr + segtext.filelen, 4096);
		while(pc < etext) {
			cput(0xf4);	// hlt
			pc++;
		}
		pc = segrodata.vaddr;
	}
	cflush();

	/* output read-only data in text segment */
	if(HEADTYPE == 8) {
		// Native Client
		sect = segrodata.sect;
		segrodata.fileoff = seek(cout, 0, 1);
	} else
		sect = segtext.sect->next;

	erodata = sect->vaddr + sect->len;
	for(v = pc; v < erodata; v += sizeof(buf)-Dbufslop) {
		if(erodata-v > sizeof(buf)-Dbufslop)
			datblk(v, sizeof(buf)-Dbufslop, 1);
		else
			datblk(v, erodata-v, 1);
	}

	switch(HEADTYPE) {
	default:
		if(iself)
			goto Elfseek;
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
	case 8:
		// Native Client only needs to round
		// text segment file address to 4096 bytes,
		// but text segment memory address rounds
		// to INITRND (65536).
		v = rnd(segrodata.fileoff+segrodata.filelen, 4096);
		seek(cout, v, 0);
		break;
	Elfseek:
	case 10:
		v = rnd(segtext.fileoff+segtext.filelen, INITRND);
		seek(cout, v, 0);
		break;
	}

	if(debug['v'])
		Bprint(&bso, "%5.2f datblk\n", cputime());
	Bflush(&bso);

	segdata.fileoff = seek(cout, 0, 1);
	for(v = 0; v < datsize; v += sizeof(buf)-Dbufslop) {
		if(datsize-v > sizeof(buf)-Dbufslop)
			datblk(v, sizeof(buf)-Dbufslop, 0);
		else
			datblk(v, datsize-v, 0);
	}

	machlink = 0;
	if(HEADTYPE == 6)
		machlink = domacholink();

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
			if(iself)
				goto Elfsym;
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
			symo = rnd(HEADR+textsize, INITRND)+rnd(datsize, INITRND)+machlink;
			break;
		Elfsym:
		case 10:
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
		if(HEADTYPE == 10 || (iself && !debug['s']))
			strnput("", INITRND-(8+symsize+lcsize)%INITRND);
		cflush();
		seek(cout, symo, 0);
		lputl(symsize);
		lputl(lcsize);
		cflush();
		if(HEADTYPE != 10 && !debug['s']) {
			seek(cout, symo+8+symsize+lcsize, 0);
			if(debug['v'])
				Bprint(&bso, "%5.2f dwarf\n", cputime());
			dwarfemitdebugsections();
		}
	}
	if(debug['v'])
		Bprint(&bso, "%5.2f headr\n", cputime());
	Bflush(&bso);
	seek(cout, 0L, 0);
	switch(HEADTYPE) {
	default:
		if(iself)
			goto Elfput;
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
		asmbmacho(symdatva, symo);
		break;

	Elfput:
		/* elf 386 */
		if(HEADTYPE == 8 || HEADTYPE == 11)
			debug['d'] = 1;

		eh = getElfEhdr();
		fo = HEADR;
		startva = INITTEXT - HEADR;
		va = startva + fo;
		w = textsize;

		/* This null SHdr must appear before all others */
		sh = newElfShdr(elfstr[ElfStrEmpty]);

		/* program header info - but not on native client */
		pph = nil;
		if(HEADTYPE != 8) {
			pph = newElfPhdr();
			pph->type = PT_PHDR;
			pph->flags = PF_R + PF_X;
			pph->off = eh->ehsize;
			pph->vaddr = INITTEXT - HEADR + pph->off;
			pph->paddr = INITTEXT - HEADR + pph->off;
			pph->align = INITRND;
		}

		if(!debug['d']) {
			/* interpreter */
			sh = newElfShdr(elfstr[ElfStrInterp]);
			sh->type = SHT_PROGBITS;
			sh->flags = SHF_ALLOC;
			sh->addralign = 1;
			switch(HEADTYPE) {
			case 7:
				elfinterp(sh, startva, linuxdynld);
				break;
			case 9:
				elfinterp(sh, startva, freebsddynld);
				break;
			}

			ph = newElfPhdr();
			ph->type = PT_INTERP;
			ph->flags = PF_R;
			phsh(ph, sh);
		}

		elfphload(&segtext);
		if(segrodata.len > 0)
			elfphload(&segrodata);
		elfphload(&segdata);

		if(!debug['s'] && HEADTYPE != 8 && HEADTYPE != 11) {
			segsym.rwx = 04;
			segsym.vaddr = symdatva;
			segsym.len = rnd(8+symsize+lcsize, INITRND);
			segsym.fileoff = symo;
			segsym.filelen = segsym.len;
			elfphload(&segsym);
		}

		/* Dynamic linking sections */
		if (!debug['d']) {	/* -d suppresses dynamic loader format */
			/* S headers for dynamic linking */
			sh = newElfShdr(elfstr[ElfStrGot]);
			sh->type = SHT_PROGBITS;
			sh->flags = SHF_ALLOC+SHF_WRITE;
			sh->entsize = 4;
			sh->addralign = 4;
			shsym(sh, lookup(".got", 0));

			sh = newElfShdr(elfstr[ElfStrGotPlt]);
			sh->type = SHT_PROGBITS;
			sh->flags = SHF_ALLOC+SHF_WRITE;
			sh->entsize = 4;
			sh->addralign = 4;
			shsym(sh, lookup(".got.plt", 0));

			dynsym = eh->shnum;
			sh = newElfShdr(elfstr[ElfStrDynsym]);
			sh->type = SHT_DYNSYM;
			sh->flags = SHF_ALLOC;
			sh->entsize = ELF32SYMSIZE;
			sh->addralign = 4;
			sh->link = dynsym+1;	// dynstr
			// sh->info = index of first non-local symbol (number of local symbols)
			shsym(sh, lookup(".dynsym", 0));

			sh = newElfShdr(elfstr[ElfStrDynstr]);
			sh->type = SHT_STRTAB;
			sh->flags = SHF_ALLOC;
			sh->addralign = 1;
			shsym(sh, lookup(".dynstr", 0));

			sh = newElfShdr(elfstr[ElfStrHash]);
			sh->type = SHT_HASH;
			sh->flags = SHF_ALLOC;
			sh->entsize = 4;
			sh->addralign = 4;
			sh->link = dynsym;
			shsym(sh, lookup(".hash", 0));

			sh = newElfShdr(elfstr[ElfStrRel]);
			sh->type = SHT_REL;
			sh->flags = SHF_ALLOC;
			sh->entsize = ELF32RELSIZE;
			sh->addralign = 4;
			sh->link = dynsym;
			shsym(sh, lookup(".rel", 0));

			/* sh and PT_DYNAMIC for .dynamic section */
			sh = newElfShdr(elfstr[ElfStrDynamic]);
			sh->type = SHT_DYNAMIC;
			sh->flags = SHF_ALLOC+SHF_WRITE;
			sh->entsize = 8;
			sh->addralign = 4;
			sh->link = dynsym+1;	// dynstr
			shsym(sh, lookup(".dynamic", 0));
			ph = newElfPhdr();
			ph->type = PT_DYNAMIC;
			ph->flags = PF_R + PF_W;
			phsh(ph, sh);

			/*
			 * Thread-local storage segment (really just size).
			 */
			if(tlsoffset != 0) {
				ph = newElfPhdr();
				ph->type = PT_TLS;
				ph->flags = PF_R;
				ph->memsz = -tlsoffset;
				ph->align = 4;
			}
		}

		ph = newElfPhdr();
		ph->type = PT_GNU_STACK;
		ph->flags = PF_W+PF_R;
		ph->align = 4;

		for(sect=segtext.sect; sect!=nil; sect=sect->next)
			elfshbits(sect);
		for(sect=segrodata.sect; sect!=nil; sect=sect->next)
			elfshbits(sect);
		for(sect=segdata.sect; sect!=nil; sect=sect->next)
			elfshbits(sect);

		if (!debug['s']) {
			fo = symo;
			w = 8;

			sh = newElfShdr(elfstr[ElfStrGosymcounts]);
			sh->type = SHT_PROGBITS;
			sh->flags = SHF_ALLOC;
			sh->off = fo;
			sh->size = w;
			sh->addralign = 1;
			sh->addr = symdatva;

			fo += w;
			w = symsize;

			sh = newElfShdr(elfstr[ElfStrGosymtab]);
			sh->type = SHT_PROGBITS;
			sh->flags = SHF_ALLOC;
			sh->off = fo;
			sh->size = w;
			sh->addralign = 1;
			sh->addr = symdatva + 8;

			fo += w;
			w = lcsize;

			sh = newElfShdr(elfstr[ElfStrGopclntab]);
			sh->type = SHT_PROGBITS;
			sh->flags = SHF_ALLOC;
			sh->off = fo;
			sh->size = w;
			sh->addralign = 1;
			sh->addr = symdatva + 8 + symsize;

			dwarfaddelfheaders();
		}

		sh = newElfShstrtab(elfstr[ElfStrShstrtab]);
		sh->type = SHT_STRTAB;
		sh->addralign = 1;
		shsym(sh, lookup(".shstrtab", 0));

		/* Main header */
		eh->ident[EI_MAG0] = '\177';
		eh->ident[EI_MAG1] = 'E';
		eh->ident[EI_MAG2] = 'L';
		eh->ident[EI_MAG3] = 'F';
		eh->ident[EI_CLASS] = ELFCLASS32;
		eh->ident[EI_DATA] = ELFDATA2LSB;
		eh->ident[EI_VERSION] = EV_CURRENT;
		switch(HEADTYPE) {
		case 8:
			eh->ident[EI_OSABI] = ELFOSABI_NACL;
			eh->ident[EI_ABIVERSION] = 7;
			eh->flags = 0x200000;	// aligned mod 32
			break;
		case 9:
			eh->ident[EI_OSABI] = 9;
			break;
		}

		eh->type = ET_EXEC;
		eh->machine = EM_386;
		eh->version = EV_CURRENT;
		eh->entry = entryvalue();

		if(pph != nil) {
			pph->filesz = eh->phnum * eh->phentsize;
			pph->memsz = pph->filesz;
		}

		seek(cout, 0, 0);
		a = 0;
		a += elfwritehdr();
		a += elfwritephdrs();
		a += elfwriteshdrs();
		cflush();
		if(a+elfwriteinterp() > ELFRESERVE)
			diag("ELFRESERVE too small: %d > %d", a, ELFRESERVE);
		break;

	case 10:
		asmbpe();
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
		ewrite(cout, buf.cbuf, n);
	cbp = buf.cbuf;
	cbc = sizeof(buf.cbuf);
}

/* Current position in file */
vlong
cpos(void)
{
	return seek(cout, 0, 1) + sizeof(buf.cbuf) - cbc;
}

void
datblk(int32 s, int32 n, int32 rodata)
{
	Prog *p;
	char *cast;
	int32 l, fl, j;
	int i, c;
	Adr *a;
	int32 base;
	
	base = INITDAT;
	if(rodata)
		base = 0;

	memset(buf.dbuf, 0, n+Dbufslop);
	for(p = datap; p != P; p = p->link) {
		a = &p->from;
		if(rodata != (a->sym->type == SRODATA))
			continue;

		l = a->sym->value + a->offset - s;
		if(l >= n)
			continue;

		c = a->scale;
		i = 0;
		if(l < 0) {
			if(l+c <= 0)
				continue;
			i = -l;
			l = 0;
		}

		curp = p;
		if(!a->sym->reachable)
			diag("unreachable symbol in datblk - %s", a->sym->name);
		if(a->sym->type == SMACHO)
			continue;

		for(j=l+(c-i)-1; j>=l; j--)
			if(buf.dbuf[j]) {
				print("%P\n", p);
				diag("multiple initialization");
				break;
			}
		switch(p->to.type) {
		case D_FCONST:
			switch(c) {
			default:
			case 4:
				fl = ieeedtof(&p->to.ieee);
				cast = (char*)&fl;
				for(; i<c; i++) {
					buf.dbuf[l] = cast[fnuxi4[i]];
					l++;
				}
				break;
			case 8:
				cast = (char*)&p->to.ieee;
				for(; i<c; i++) {
					buf.dbuf[l] = cast[fnuxi8[i]];
					l++;
				}
				break;
			}
			break;

		case D_SCONST:
			for(; i<c; i++) {
				buf.dbuf[l] = p->to.scon[i];
				l++;
			}
			break;

		default:
			fl = p->to.offset;
			if(p->to.type == D_SIZE)
				fl += p->to.sym->size;
			if(p->to.type == D_ADDR) {
				if(p->to.index != D_STATIC && p->to.index != D_EXTERN)
					diag("DADDR type%P", p);
				if(p->to.sym) {
					if(p->to.sym->type == Sxxx) {
						cursym = p->from.sym;
						diag("missing symbol %s", p->to.sym->name);
						cursym = nil;
					}
					fl += p->to.sym->value;
					if(p->to.sym->type != STEXT && p->to.sym->type != SRODATA)
						fl += INITDAT;
				}
			}
			cast = (char*)&fl;
			switch(c) {
			default:
				diag("bad nuxi %d %d\n%P", c, i, curp);
				break;
			case 1:
				for(; i<c; i++) {
					buf.dbuf[l] = cast[inuxi1[i]];
					l++;
				}
				break;
			case 2:
				for(; i<c; i++) {
					buf.dbuf[l] = cast[inuxi2[i]];
					l++;
				}
				break;
			case 4:
				for(; i<c; i++) {
					buf.dbuf[l] = cast[inuxi4[i]];
					l++;
				}
				break;
			}
			break;
		}
	}

	ewrite(cout, buf.dbuf, n);
	if(!debug['a'])
		return;

	/*
	 * a second pass just to print the asm
	 */
	for(p = datap; p != P; p = p->link) {
		a = &p->from;
		if(rodata != (a->sym->type == SRODATA))
			continue;

		l = a->sym->value + a->offset - s;
		if(l < 0 || l >= n)
			continue;

		c = a->scale;
		i = 0;

		switch(p->to.type) {
		case D_FCONST:
			switch(c) {
			default:
			case 4:
				fl = ieeedtof(&p->to.ieee);
				cast = (char*)&fl;
				Bprint(&bso, pcstr, l+s+base);
				for(j=0; j<c; j++)
					Bprint(&bso, "%.2ux", cast[fnuxi4[j]] & 0xff);
				Bprint(&bso, "\t%P\n", p);
				break;
			case 8:
				cast = (char*)&p->to.ieee;
				Bprint(&bso, pcstr, l+s+base);
				for(j=0; j<c; j++)
					Bprint(&bso, "%.2ux", cast[fnuxi8[j]] & 0xff);
				Bprint(&bso, "\t%P\n", p);
				break;
			}
			break;

		case D_SCONST:
			Bprint(&bso, pcstr, l+s+base);
			for(j=0; j<c; j++)
				Bprint(&bso, "%.2ux", p->to.scon[j] & 0xff);
			Bprint(&bso, "\t%P\n", p);
			break;

		default:
			fl = p->to.offset;
			if(p->to.type == D_SIZE)
				fl += p->to.sym->size;
			if(p->to.type == D_ADDR) {
				if(p->to.index != D_STATIC && p->to.index != D_EXTERN)
					diag("DADDR type%P", p);
				if(p->to.sym) {
					fl += p->to.sym->value;
					if(p->to.sym->type != STEXT && p->to.sym->type != SRODATA)
						fl += INITDAT;
				}
			}
			cast = (char*)&fl;
			switch(c) {
			default:
				diag("bad nuxi %d %d\n%P", c, i, p);
				break;
			case 1:
				Bprint(&bso, pcstr, l+s+base);
				for(j=0; j<c; j++)
					Bprint(&bso, "%.2ux", cast[inuxi1[j]] & 0xff);
				Bprint(&bso, "\t%P\n", p);
				break;
			case 2:
				Bprint(&bso, pcstr, l+s+base);
				for(j=0; j<c; j++)
					Bprint(&bso, "%.2ux", cast[inuxi2[j]] & 0xff);
				Bprint(&bso, "\t%P\n", p);
				break;
			case 4:
				Bprint(&bso, pcstr, l+s+base);
				for(j=0; j<c; j++)
					Bprint(&bso, "%.2ux", cast[inuxi4[j]] & 0xff);
				Bprint(&bso, "\t%P\n", p);
				break;
			}
			break;
		}
	}
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
