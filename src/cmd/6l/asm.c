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
#include	"../ld/lib.h"
#include	"../ld/elf.h"
#include	"../ld/macho.h"

#define	Dbufslop	100

#define PADDR(a)	((uint32)(a) & ~0x80000000)

char linuxdynld[] = "/lib64/ld-linux-x86-64.so.2";
char freebsddynld[] = "/libexec/ld-elf.so.1";

char	zeroes[32];
Prog*	datsort(Prog *l);

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

vlong
addstring(Sym *s, char *str)
{
	int n, m;
	vlong r;
	Prog *p;

	if(s->type == 0)
		s->type = SDATA;
	s->reachable = 1;
	r = s->value;
	n = strlen(str)+1;
	while(n > 0) {
		m = n;
		if(m > sizeof(p->to.scon))
			m = sizeof(p->to.scon);
		p = newdata(s, s->value, m, D_EXTERN);
		p->to.type = D_SCONST;
		memmove(p->to.scon, str, m);
		s->value += m;
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
	r = s->value;
	p = newdata(s, s->value, wid, D_EXTERN);
	s->value += wid;
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
	enum { Ptrsize = 8 };

	if(s->type == 0)
		s->type = SDATA;
	s->reachable = 1;
	r = s->value;
	p = newdata(s, s->value, Ptrsize, D_EXTERN);
	s->value += Ptrsize;
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
	enum { Ptrsize = 8 };

	if(s->type == 0)
		s->type = SDATA;
	s->reachable = 1;
	r = s->value;
	p = newdata(s, s->value, Ptrsize, D_EXTERN);
	s->value += Ptrsize;
	p->to.type = D_SIZE;
	p->to.index = D_EXTERN;
	p->to.offset = 0;
	p->to.sym = t;
	return r;
}

vlong
datoff(vlong addr)
{
	if(addr >= INITDAT)
		return addr - INITDAT + rnd(HEADR+textsize, INITRND);
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
	ElfStrRela,
	ElfStrText,
	ElfStrData,
	ElfStrBss,
	ElfStrGosymtab,
	ElfStrGopclntab,
	ElfStrShstrtab,
	ElfStrSymtab,
	ElfStrStrtab,
	NElfStr
};

vlong elfstr[NElfStr];

static int
needlib(char *name)
{
	char *p;
	Sym *s;

	/* reuse hash code in symbol table */
	p = smprint(".elfload.%s", name);
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

	if(HEADTYPE != 7 && HEADTYPE != 9)
		return;

	/* predefine strings we need for section headers */
	shstrtab = lookup(".shstrtab", 0);
	elfstr[ElfStrEmpty] = addstring(shstrtab, "");
	elfstr[ElfStrText] = addstring(shstrtab, ".text");
	elfstr[ElfStrData] = addstring(shstrtab, ".data");
	elfstr[ElfStrBss] = addstring(shstrtab, ".bss");
	if(!debug['s']) {
		elfstr[ElfStrGosymtab] = addstring(shstrtab, ".gosymtab");
		elfstr[ElfStrGopclntab] = addstring(shstrtab, ".gopclntab");
		if(debug['e']) {
			elfstr[ElfStrSymtab] = addstring(shstrtab, ".symtab");
			elfstr[ElfStrStrtab] = addstring(shstrtab, ".strtab");
		}
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
		elfstr[ElfStrRela] = addstring(shstrtab, ".rela");

		/* dynamic symbol table - first entry all zeros */
		s = lookup(".dynsym", 0);
		s->type = SDATA;
		s->reachable = 1;
		s->value += ELF64SYMSIZE;

		/* dynamic string table */
		s = lookup(".dynstr", 0);
		addstring(s, "");
		dynstr = s;

		/* relocation table */
		s = lookup(".rela", 0);
		s->reachable = 1;
		s->type = SDATA;

		/* global offset table */
		s = lookup(".got", 0);
		s->reachable = 1;
		s->type = SDATA;

		/* got.plt - ??? */
		s = lookup(".got.plt", 0);
		s->reachable = 1;
		s->type = SDATA;

		/* define dynamic elf table */
		s = lookup(".dynamic", 0);
		dynamic = s;

		/*
		 * relocation entries for dynld symbols
		 */
		nsym = 1;	// sym 0 is reserved
		for(h=0; h<NHASH; h++) {
			for(s=hash[h]; s!=S; s=s->link) {
				if(!s->reachable || (s->type != SDATA && s->type != SBSS) || s->dynldname == nil)
					continue;

				d = lookup(".rela", 0);
				addaddr(d, s);
				adduint64(d, ELF64_R_INFO(nsym, R_X86_64_64));
				adduint64(d, 0);
				nsym++;

				d = lookup(".dynsym", 0);
				adduint32(d, addstring(lookup(".dynstr", 0), s->dynldname));
				t = STB_GLOBAL << 4;
				t |= STT_OBJECT;	// works for func too, empirically
				adduint8(d, t);
				adduint8(d, 0);	/* reserved */
				adduint16(d, SHN_UNDEF);	/* section where symbol is defined */
				adduint64(d, 0);	/* value */
				adduint64(d, 0);	/* size of object */

				if(needlib(s->dynldlib))
					elfwritedynent(dynamic, DT_NEEDED, addstring(dynstr, s->dynldlib));
			}
		}

		/*
		 * hash table.
		 * only entries that other objects need to find when
		 * linking us need to be in the table.  right now that is
		 * no entries.
		 *
		 * freebsd insists on having chains enough for all
		 * the local symbols, though.  for now, we just lay
		 * down a trivial hash table with 1 bucket and a long chain,
		 * because no one is actually looking for our symbols.
		 */
		s = lookup(".hash", 0);
		s->type = SDATA;	// TODO: rodata
		s->reachable = 1;
		adduint32(s, 1);	// nbucket
		adduint32(s, nsym);	// nchain
		adduint32(s, nsym-1);	// bucket 0
		adduint32(s, 0);	// chain 0
		for(h=1; h<nsym; h++)	// chain nsym-1 -> nsym-2 -> ... -> 2 -> 1 -> 0
			adduint32(s, h-1);

		/*
		 * .dynamic table
		 */
		s = dynamic;
		elfwritedynentsym(s, DT_HASH, lookup(".hash", 0));
		elfwritedynentsym(s, DT_SYMTAB, lookup(".dynsym", 0));
		elfwritedynent(s, DT_SYMENT, ELF64SYMSIZE);
		elfwritedynentsym(s, DT_STRTAB, lookup(".dynstr", 0));
		elfwritedynentsymsize(s, DT_STRSZ, lookup(".dynstr", 0));
		elfwritedynentsym(s, DT_RELA, lookup(".rela", 0));
		elfwritedynentsymsize(s, DT_RELASZ, lookup(".rela", 0));
		elfwritedynent(s, DT_RELAENT, ELF64RELASIZE);
		elfwritedynent(s, DT_NULL, 0);
	}
}

void
shsym(ElfShdr *sh, Sym *s)
{
	sh->addr = symaddr(s);
	sh->off = datoff(sh->addr);
	sh->size = s->size;
}

void
phsh(ElfPhdr *ph, ElfShdr *sh)
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
	uchar *op1;
	vlong vl, va, startva, fo, w, symo, elfsymo, elfstro, elfsymsize, machlink;
	vlong symdatva = SYMDATVA;
	ElfEhdr *eh;
	ElfPhdr *ph, *pph;
	ElfShdr *sh;

	if(debug['v'])
		Bprint(&bso, "%5.2f asmb\n", cputime());
	Bflush(&bso);

	elftextsh = 0;
	elfsymsize = 0;
	elfstro = 0;
	elfsymo = 0;
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
	case 9:
		debug['8'] = 1;	/* 64-bit addresses */
		v = rnd(HEADR+textsize, INITRND);
		seek(cout, v, 0);
		
		/* index of elf text section; needed by asmelfsym, double-checked below */
		/* debug['d'] causes 8 extra sections before the .text section */
		elftextsh = 1;
		if(!debug['d'])
			elftextsh += 8;
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

	datap = datsort(datap);
	for(v = 0; v < datsize; v += sizeof(buf)-Dbufslop) {
		if(datsize-v > sizeof(buf)-Dbufslop)
			datblk(v, sizeof(buf)-Dbufslop);
		else
			datblk(v, datsize-v);
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
		case 2:
		case 5:
			debug['s'] = 1;
			symo = HEADR+textsize+datsize;
			break;
		case 6:
			symo = rnd(HEADR+textsize, INITRND)+rnd(datsize, INITRND)+machlink;
			break;
		case 7:
		case 9:
			symo = rnd(HEADR+textsize, INITRND)+datsize;
			symo = rnd(symo, INITRND);
			break;
		}
		/*
		 * the symbol information is stored as
		 *	32-bit symbol table size
		 *	32-bit line number table size
		 *	symbol table
		 *	line number table
		 */
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
		if(!debug['s'] && debug['e']) {
			elfsymo = symo+8+symsize+lcsize;
			seek(cout, elfsymo, 0);
			asmelfsym();
			cflush();
			elfstro = seek(cout, 0, 1);
			elfsymsize = elfstro - elfsymo;
			write(cout, elfstrdat, elfstrsize);
		}		
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
	case 6:
		asmbmacho(symdatva, symo);
		break;
	case 7:
	case 9:
		/* elf amd-64 */

		eh = getElfEhdr();
		fo = HEADR;
		startva = INITTEXT - HEADR;
		va = startva + fo;
		w = textsize;

		/* This null SHdr must appear before all others */
		sh = newElfShdr(elfstr[ElfStrEmpty]);

		/* program header info */
		pph = newElfPhdr();
		pph->type = PT_PHDR;
		pph->flags = PF_R + PF_X;
		pph->off = eh->ehsize;
		pph->vaddr = INITTEXT - HEADR + pph->off;
		pph->paddr = INITTEXT - HEADR + pph->off;
		pph->align = INITRND;

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

		ph = newElfPhdr();
		ph->type = PT_LOAD;
		ph->flags = PF_X+PF_R;
		ph->vaddr = va;
		ph->paddr = va;
		ph->off = fo;
		ph->filesz = w;
		ph->memsz = w;
		ph->align = INITRND;

		fo = rnd(fo+w, INITRND);
		va = rnd(va+w, INITRND);
		w = datsize;

		ph = newElfPhdr();
		ph->type = PT_LOAD;
		ph->flags = PF_W+PF_R;
		ph->off = fo;
		ph->vaddr = va;
		ph->paddr = va;
		ph->filesz = w;
		ph->memsz = w+bsssize;
		ph->align = INITRND;

		if(!debug['s']) {
			ph = newElfPhdr();
			ph->type = PT_LOAD;
			ph->flags = PF_W+PF_R;
			ph->off = symo;
			ph->vaddr = symdatva;
			ph->paddr = symdatva;
			ph->filesz = 8+symsize+lcsize;
			ph->memsz = 8+symsize+lcsize;
			ph->align = INITRND;
		}

		/* Dynamic linking sections */
		if (!debug['d']) {	/* -d suppresses dynamic loader format */
			/* S headers for dynamic linking */
			sh = newElfShdr(elfstr[ElfStrGot]);
			sh->type = SHT_PROGBITS;
			sh->flags = SHF_ALLOC+SHF_WRITE;
			sh->entsize = 8;
			sh->addralign = 8;
			shsym(sh, lookup(".got", 0));

			sh = newElfShdr(elfstr[ElfStrGotPlt]);
			sh->type = SHT_PROGBITS;
			sh->flags = SHF_ALLOC+SHF_WRITE;
			sh->entsize = 8;
			sh->addralign = 8;
			shsym(sh, lookup(".got.plt", 0));

			dynsym = eh->shnum;
			sh = newElfShdr(elfstr[ElfStrDynsym]);
			sh->type = SHT_DYNSYM;
			sh->flags = SHF_ALLOC;
			sh->entsize = ELF64SYMSIZE;
			sh->addralign = 8;
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
			sh->addralign = 8;
			sh->link = dynsym;
			shsym(sh, lookup(".hash", 0));

			sh = newElfShdr(elfstr[ElfStrRela]);
			sh->type = SHT_RELA;
			sh->flags = SHF_ALLOC;
			sh->entsize = ELF64RELASIZE;
			sh->addralign = 8;
			sh->link = dynsym;
			shsym(sh, lookup(".rela", 0));

			/* sh and PT_DYNAMIC for .dynamic section */
			sh = newElfShdr(elfstr[ElfStrDynamic]);
			sh->type = SHT_DYNAMIC;
			sh->flags = SHF_ALLOC+SHF_WRITE;
			sh->entsize = 16;
			sh->addralign = 8;
			sh->link = dynsym+1;	// dynstr
			shsym(sh, lookup(".dynamic", 0));
			ph = newElfPhdr();
			ph->type = PT_DYNAMIC;
			ph->flags = PF_R + PF_W;
			phsh(ph, sh);
		}

		ph = newElfPhdr();
		ph->type = PT_GNU_STACK;
		ph->flags = PF_W+PF_R;
		ph->align = 8;

		fo = ELFRESERVE;
		va = startva + fo;
		w = textsize;

		if(elftextsh != eh->shnum)
			diag("elftextsh = %d, want %d", elftextsh, eh->shnum);
		sh = newElfShdr(elfstr[ElfStrText]);
		sh->type = SHT_PROGBITS;
		sh->flags = SHF_ALLOC+SHF_EXECINSTR;
		sh->addr = va;
		sh->off = fo;
		sh->size = w;
		sh->addralign = 8;

		fo = rnd(fo+w, INITRND);
		va = rnd(va+w, INITRND);
		w = datsize;

		sh = newElfShdr(elfstr[ElfStrData]);
		sh->type = SHT_PROGBITS;
		sh->flags = SHF_WRITE+SHF_ALLOC;
		sh->addr = va;
		sh->off = fo;
		sh->size = w;
		sh->addralign = 8;

		fo += w;
		va += w;
		w = bsssize;

		sh = newElfShdr(elfstr[ElfStrBss]);
		sh->type = SHT_NOBITS;
		sh->flags = SHF_WRITE+SHF_ALLOC;
		sh->addr = va;
		sh->off = fo;
		sh->size = w;
		sh->addralign = 8;

		if (!debug['s']) {
			fo = symo+8;
			w = symsize;

			sh = newElfShdr(elfstr[ElfStrGosymtab]);
			sh->type = SHT_PROGBITS;
			sh->off = fo;
			sh->size = w;
			sh->addralign = 1;

			fo += w;
			w = lcsize;

			sh = newElfShdr(elfstr[ElfStrGopclntab]);
			sh->type = SHT_PROGBITS;
			sh->off = fo;
			sh->size = w;
			sh->addralign = 1;
			
			if(debug['e']) {
				sh = newElfShdr(elfstr[ElfStrSymtab]);
				sh->type = SHT_SYMTAB;
				sh->off = elfsymo;
				sh->size = elfsymsize;
				sh->addralign = 8;
				sh->entsize = 24;
				sh->link = eh->shnum;	// link to strtab
			
				sh = newElfShdr(elfstr[ElfStrStrtab]);
				sh->type = SHT_STRTAB;
				sh->off = elfstro;
				sh->size = elfstrsize;
				sh->addralign = 1;
			}
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
		if(HEADTYPE == 9)
			eh->ident[EI_OSABI] = 9;
		eh->ident[EI_CLASS] = ELFCLASS64;
		eh->ident[EI_DATA] = ELFDATA2LSB;
		eh->ident[EI_VERSION] = EV_CURRENT;

		eh->type = ET_EXEC;
		eh->machine = EM_X86_64;
		eh->version = EV_CURRENT;
		eh->entry = entryvalue();

		pph->filesz = eh->phnum * eh->phentsize;
		pph->memsz = pph->filesz;

		seek(cout, 0, 0);
		a = 0;
		a += elfwritehdr();
		a += elfwritephdrs();
		a += elfwriteshdrs();
		cflush();
		if(a+elfwriteinterp() > ELFRESERVE)
			diag("ELFRESERVE too small: %d > %d", a, ELFRESERVE);
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

/*
 * divide-and-conquer list-link
 * sort of Prog* structures.
 * Used for the data block.
 */
int
datcmp(Prog *p1, Prog *p2)
{
	vlong v1, v2;

	v1 = p1->from.offset;
	v2 = p2->from.offset;
	if(v1 > v2)
		return +1;
	if(v1 < v2)
		return -1;
	return 0;
}

Prog*
dsort(Prog *l)
{
	Prog *l1, *l2, *le;

	if(l == 0 || l->link == 0)
		return l;

	l1 = l;
	l2 = l;
	for(;;) {
		l2 = l2->link;
		if(l2 == 0)
			break;
		l2 = l2->link;
		if(l2 == 0)
			break;
		l1 = l1->link;
	}

	l2 = l1->link;
	l1->link = 0;
	l1 = dsort(l);
	l2 = dsort(l2);

	/* set up lead element */
	if(datcmp(l1, l2) < 0) {
		l = l1;
		l1 = l1->link;
	} else {
		l = l2;
		l2 = l2->link;
	}
	le = l;

	for(;;) {
		if(l1 == 0) {
			while(l2) {
				le->link = l2;
				le = l2;
				l2 = l2->link;
			}
			le->link = 0;
			break;
		}
		if(l2 == 0) {
			while(l1) {
				le->link = l1;
				le = l1;
				l1 = l1->link;
			}
			break;
		}
		if(datcmp(l1, l2) < 0) {
			le->link = l1;
			le = l1;
			l1 = l1->link;
		} else {
			le->link = l2;
			le = l2;
			l2 = l2->link;
		}
	}
	le->link = 0;
	return l;
}

static Prog *datp;

Prog*
datsort(Prog *l)
{
	Prog *p;
	Adr *a;

	for(p = l; p != P; p = p->link) {
		a = &p->from;
		a->offset += a->sym->value;
	}
	datp = dsort(l);
	return datp;
}

void
datblk(int32 s, int32 n)
{
	Prog *p;
	uchar *cast;
	int32 l, fl, j;
	vlong o;
	int i, c;
	Adr *a;

	for(p = datp; p != P; p = p->link) {
		a = &p->from;
		l = a->offset - s;
		if(l+a->scale < 0)
			continue;
		datp = p;
		break;
	}

	memset(buf.dbuf, 0, n+Dbufslop);
	for(p = datp; p != P; p = p->link) {
		a = &p->from;

		l = a->offset - s;
		if(l >= n)
			break;

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
				for(; i<c; i++) {
					buf.dbuf[l] = cast[fnuxi4[i]];
					l++;
				}
				break;
			case 8:
				cast = (uchar*)&p->to.ieee;
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
			o = p->to.offset;
			if(p->to.type == D_SIZE)
				o += p->to.sym->size;
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
			case 8:
				cast = (uchar*)&o;
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
	if(!debug['a'])
		return;

	/*
	 * a second pass just to print the asm
	 */
	for(p = datap; p != P; p = p->link) {
		a = &p->from;

		l = a->offset - s;
		if(l >= n)
			continue;

		c = a->scale;
		i = 0;
		if(l < 0)
			continue;

		if(a->sym->type == SMACHO)
			continue;

		switch(p->to.type) {
		case D_FCONST:
			switch(c) {
			default:
			case 4:
				fl = ieeedtof(&p->to.ieee);
				cast = (uchar*)&fl;
				outa(c, cast, fnuxi4, l+s+INITDAT);
				break;
			case 8:
				cast = (uchar*)&p->to.ieee;
				outa(c, cast, fnuxi8, l+s+INITDAT);
				break;
			}
			break;

		case D_SCONST:
			outa(c, (uchar*)p->to.scon, nil, l+s+INITDAT);
			break;

		default:
			o = p->to.offset;
			if(p->to.type == D_SIZE)
				o += p->to.sym->size;
			if(p->to.type == D_ADDR) {
				if(p->to.sym) {
					o += p->to.sym->value;
					if(p->to.sym->type != STEXT && p->to.sym->type != SUNDEF)
						o += INITDAT;
				}
			}
			fl = o;
			cast = (uchar*)&fl;
			switch(c) {
			case 1:
				outa(c, cast, inuxi1, l+s+INITDAT);
				break;
			case 2:
				outa(c, cast, inuxi2, l+s+INITDAT);
				break;
			case 4:
				outa(c, cast, inuxi4, l+s+INITDAT);
				break;
			case 8:
				cast = (uchar*)&o;
				outa(c, cast, inuxi8, l+s+INITDAT);
				break;
			}
			break;
		}
	}
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

