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

// Writing object files.

#include	"l.h"
#include	"../ld/lib.h"
#include	"../ld/elf.h"
#include	"../ld/dwarf.h"
#include	"../ld/macho.h"

#define	Dbufslop	100

#define PADDR(a)	((uint32)(a) & ~0x80000000)

char linuxdynld[] = "/lib64/ld-linux-x86-64.so.2";
char freebsddynld[] = "/libexec/ld-elf.so.1";

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
	if(s->type != STEXT)
		diag("entry not text: %s", s->name);
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

vlong
datoff(vlong addr)
{
	if(addr >= segdata.vaddr)
		return addr - segdata.vaddr + segdata.fileoff;
	if(addr >= segtext.vaddr)
		return addr - segtext.vaddr + segtext.fileoff;
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
	ElfStrGosymcounts,
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
	shstrtab->type = SELFDATA;
	shstrtab->reachable = 1;

	elfstr[ElfStrEmpty] = addstring(shstrtab, "");
	elfstr[ElfStrText] = addstring(shstrtab, ".text");
	elfstr[ElfStrData] = addstring(shstrtab, ".data");
	elfstr[ElfStrBss] = addstring(shstrtab, ".bss");
	addstring(shstrtab, ".elfdata");
	addstring(shstrtab, ".rodata");
	if(!debug['s']) {
		elfstr[ElfStrGosymcounts] = addstring(shstrtab, ".gosymcounts");
		elfstr[ElfStrGosymtab] = addstring(shstrtab, ".gosymtab");
		elfstr[ElfStrGopclntab] = addstring(shstrtab, ".gopclntab");
		elfstr[ElfStrSymtab] = addstring(shstrtab, ".symtab");
		elfstr[ElfStrStrtab] = addstring(shstrtab, ".strtab");
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
		elfstr[ElfStrRela] = addstring(shstrtab, ".rela");

		/* dynamic symbol table - first entry all zeros */
		s = lookup(".dynsym", 0);
		s->type = SELFDATA;
		s->reachable = 1;
		s->size += ELF64SYMSIZE;

		/* dynamic string table */
		s = lookup(".dynstr", 0);
		s->type = SELFDATA;
		s->reachable = 1;
		addstring(s, "");
		dynstr = s;

		/* relocation table */
		s = lookup(".rela", 0);
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
		
		/* hash */
		s = lookup(".hash", 0);
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
					d = lookup(".rela", 0);
					addaddr(d, s);
					adduint64(d, ELF64_R_INFO(nsym, R_X86_64_64));
					adduint64(d, 0);
				}

				nsym++;

				d = lookup(".dynsym", 0);
				adduint32(d, addstring(lookup(".dynstr", 0), s->dynimpname));
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

				/* value */
				if(!s->dynexport)
					adduint64(d, 0);
				else
					addaddr(d, s);

				/* size of object */
				adduint64(d, 0);

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
		elfwritedynent(s, DT_SYMENT, ELF64SYMSIZE);
		elfwritedynentsym(s, DT_STRTAB, lookup(".dynstr", 0));
		elfwritedynentsymsize(s, DT_STRSZ, lookup(".dynstr", 0));
		elfwritedynentsym(s, DT_RELA, lookup(".rela", 0));
		elfwritedynentsymsize(s, DT_RELASZ, lookup(".rela", 0));
		elfwritedynent(s, DT_RELAENT, ELF64RELASIZE);
		if(rpath)
			elfwritedynent(s, DT_RUNPATH, addstring(dynstr, rpath));
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
	int32 v, magic;
	int a, dynsym;
	vlong vl, va, startva, fo, w, symo, elfsymo, elfstro, elfsymsize, machlink;
	vlong symdatva = SYMDATVA;
	ElfEhdr *eh;
	ElfPhdr *ph, *pph;
	ElfShdr *sh;
	Section *sect;

	if(debug['v'])
		Bprint(&bso, "%5.2f asmb\n", cputime());
	Bflush(&bso);

	segtext.fileoff = 0;
	elftextsh = 0;
	elfsymsize = 0;
	elfstro = 0;
	elfsymo = 0;
	seek(cout, HEADR, 0);
	pc = INITTEXT;
	codeblk(pc, segtext.sect->len);
	pc += segtext.sect->len;

	/* output read-only data in text segment */
	sect = segtext.sect->next;
	datblk(pc, sect->vaddr + sect->len - pc);

	switch(HEADTYPE) {
	default:
		diag("unknown header type %d", HEADTYPE);
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
		/* !debug['d'] causes 8 extra sections before the .text section */
		elftextsh = 1;
		if(!debug['d'])
			elftextsh += 8;
		break;
	}

	if(debug['v'])
		Bprint(&bso, "%5.2f datblk\n", cputime());
	Bflush(&bso);

	segdata.fileoff = seek(cout, 0, 1);
	datblk(INITDAT, segdata.filelen);

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
			symo = HEADR+textsize+segdata.filelen;
			break;
		case 6:
			symo = rnd(HEADR+textsize, INITRND)+rnd(segdata.filelen, INITRND)+machlink;
			break;
		case 7:
		case 9:
			symo = rnd(HEADR+textsize, INITRND)+segdata.filelen;
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
		if(!debug['s'])
			strnput("", INITRND-(8+symsize+lcsize)%INITRND);
		cflush();
		seek(cout, symo, 0);
		lputl(symsize);
		lputl(lcsize);
		cflush();
		if(!debug['s']) {
			elfsymo = symo+8+symsize+lcsize;
			seek(cout, elfsymo, 0);
			asmelfsym();
			cflush();
			elfstro = seek(cout, 0, 1);
			elfsymsize = elfstro - elfsymo;
			ewrite(cout, elfstrdat, elfstrsize);

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
	case 2:	/* plan9 */
		magic = 4*26*26+7;
		magic |= 0x00008000;		/* fat header */
		lputb(magic);			/* magic */
		lputb(textsize);			/* sizes */
		lputb(segdata.filelen);
		lputb(segdata.len - segdata.filelen);
		lputb(symsize);			/* nsyms */
		vl = entryvalue();
		lputb(PADDR(vl));		/* va of entry */
		lputb(spsize);			/* sp offsets */
		lputb(lcsize);			/* line offsets */
		vputb(vl);			/* va of entry */
		break;
	case 3:	/* plan9 */
		magic = 4*26*26+7;
		lputb(magic);			/* magic */
		lputb(textsize);		/* sizes */
		lputb(segdata.filelen);
		lputb(segdata.len - segdata.filelen);
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

		elfphload(&segtext);
		elfphload(&segdata);

		if(!debug['s']) {
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
			
			/*
			 * Thread-local storage segment (really just size).
			 */
			if(tlsoffset != 0) {
				ph = newElfPhdr();
				ph->type = PT_TLS;
				ph->flags = PF_R;
				ph->memsz = -tlsoffset;
				ph->align = 8;
			}
		}

		ph = newElfPhdr();
		ph->type = PT_GNU_STACK;
		ph->flags = PF_W+PF_R;
		ph->align = 8;

		if(elftextsh != eh->shnum)
			diag("elftextsh = %d, want %d", elftextsh, eh->shnum);
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
