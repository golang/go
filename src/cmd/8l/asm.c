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
char openbsddynld[] = "/usr/libexec/ld.so";
char netbsddynld[] = "/usr/libexec/ld.elf_so";

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
	if(s->type != STEXT)
		diag("entry not text: %s", s->name);
	return s->value;
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
	ElfStrRel,
	ElfStrText,
	ElfStrData,
	ElfStrBss,
	ElfStrShstrtab,
	ElfStrSymtab,
	ElfStrStrtab,
	ElfStrRelPlt,
	ElfStrPlt,
	ElfStrGnuVersion,
	ElfStrGnuVersionR,
	ElfStrNoteNetbsdIdent,
	NElfStr
};

vlong elfstr[NElfStr];

static int
needlib(char *name)
{
	char *p;
	Sym *s;

	if(*name == '\0')
		return 0;

	/* reuse hash code in symbol table */
	p = smprint(".dynlib.%s", name);
	s = lookup(p, 0);
	if(s->type == 0) {
		s->type = 100;	// avoid SDATA, etc.
		return 1;
	}
	return 0;
}

int	nelfsym = 1;

static void	addpltsym(Sym*);
static void	addgotsym(Sym*);

void
adddynrel(Sym *s, Reloc *r)
{
	Sym *targ, *rel, *got;

	targ = r->sym;
	cursym = s;

	switch(r->type) {
	default:
		if(r->type >= 256) {
			diag("unexpected relocation type %d", r->type);
			return;
		}
		break;

	// Handle relocations found in ELF object files.
	case 256 + R_386_PC32:
		if(targ->dynimpname != nil && !targ->dynexport)
			diag("unexpected R_386_PC32 relocation for dynamic symbol %s", targ->name);
		if(targ->type == 0 || targ->type == SXREF)
			diag("unknown symbol %s in pcrel", targ->name);
		r->type = D_PCREL;
		r->add += 4;
		return;

	case 256 + R_386_PLT32:
		r->type = D_PCREL;
		r->add += 4;
		if(targ->dynimpname != nil && !targ->dynexport) {
			addpltsym(targ);
			r->sym = lookup(".plt", 0);
			r->add += targ->plt;
		}
		return;		
	
	case 256 + R_386_GOT32:
		if(targ->dynimpname == nil || targ->dynexport) {
			// have symbol
			// turn MOVL of GOT entry into LEAL of symbol itself
			if(r->off < 2 || s->p[r->off-2] != 0x8b) {
				diag("unexpected GOT reloc for non-dynamic symbol %s", targ->name);
				return;
			}
			s->p[r->off-2] = 0x8d;
			r->type = D_GOTOFF;
			return;
		}
		addgotsym(targ);
		r->type = D_CONST;	// write r->add during relocsym
		r->sym = S;
		r->add += targ->got;
		return;
	
	case 256 + R_386_GOTOFF:
		r->type = D_GOTOFF;
		return;
	
	case 256 + R_386_GOTPC:
		r->type = D_PCREL;
		r->sym = lookup(".got", 0);
		r->add += 4;
		return;

	case 256 + R_386_32:
		if(targ->dynimpname != nil && !targ->dynexport)
			diag("unexpected R_386_32 relocation for dynamic symbol %s", targ->name);
		r->type = D_ADDR;
		return;
	
	case 512 + MACHO_GENERIC_RELOC_VANILLA*2 + 0:
		r->type = D_ADDR;
		if(targ->dynimpname != nil && !targ->dynexport)
			diag("unexpected reloc for dynamic symbol %s", targ->name);
		return;
	
	case 512 + MACHO_GENERIC_RELOC_VANILLA*2 + 1:
		if(targ->dynimpname != nil && !targ->dynexport) {
			addpltsym(targ);
			r->sym = lookup(".plt", 0);
			r->add = targ->plt;
			r->type = D_PCREL;
			return;
		}
		r->type = D_PCREL;
		return;
	
	case 512 + MACHO_FAKE_GOTPCREL:
		if(targ->dynimpname == nil || targ->dynexport) {
			// have symbol
			// turn MOVL of GOT entry into LEAL of symbol itself
			if(r->off < 2 || s->p[r->off-2] != 0x8b) {
				diag("unexpected GOT reloc for non-dynamic symbol %s", targ->name);
				return;
			}
			s->p[r->off-2] = 0x8d;
			r->type = D_PCREL;
			return;
		}
		addgotsym(targ);
		r->sym = lookup(".got", 0);
		r->add += targ->got;
		r->type = D_PCREL;
		return;
	}
	
	// Handle references to ELF symbols from our own object files.
	if(targ->dynimpname == nil || targ->dynexport)
		return;

	switch(r->type) {
	case D_PCREL:
		addpltsym(targ);
		r->sym = lookup(".plt", 0);
		r->add = targ->plt;
		return;
	
	case D_ADDR:
		if(s->type != SDATA)
			break;
		if(iself) {
			adddynsym(targ);
			rel = lookup(".rel", 0);
			addaddrplus(rel, s, r->off);
			adduint32(rel, ELF32_R_INFO(targ->dynid, R_386_32));
			r->type = D_CONST;	// write r->add during relocsym
			r->sym = S;
			return;
		}
		if(HEADTYPE == Hdarwin && s->size == PtrSize && r->off == 0) {
			// Mach-O relocations are a royal pain to lay out.
			// They use a compact stateful bytecode representation
			// that is too much bother to deal with.
			// Instead, interpret the C declaration
			//	void *_Cvar_stderr = &stderr;
			// as making _Cvar_stderr the name of a GOT entry
			// for stderr.  This is separate from the usual GOT entry,
			// just in case the C code assigns to the variable,
			// and of course it only works for single pointers,
			// but we only need to support cgo and that's all it needs.
			adddynsym(targ);
			got = lookup(".got", 0);
			s->type = got->type | SSUB;
			s->outer = got;
			s->sub = got->sub;
			got->sub = s;
			s->value = got->size;
			adduint32(got, 0);
			adduint32(lookup(".linkedit.got", 0), targ->dynid);
			r->type = 256;	// ignore during relocsym
			return;
		}
		break;
	}
	
	cursym = s;
	diag("unsupported relocation for dynamic symbol %s (type=%d stype=%d)", targ->name, r->type, targ->type);
}

static void
elfsetupplt(void)
{
	Sym *plt, *got;
	
	plt = lookup(".plt", 0);
	got = lookup(".got.plt", 0);
	if(plt->size == 0) {
		// pushl got+4
		adduint8(plt, 0xff);
		adduint8(plt, 0x35);
		addaddrplus(plt, got, 4);
		
		// jmp *got+8
		adduint8(plt, 0xff);
		adduint8(plt, 0x25);
		addaddrplus(plt, got, 8);

		// zero pad
		adduint32(plt, 0);
		
		// assume got->size == 0 too
		addaddrplus(got, lookup(".dynamic", 0), 0);
		adduint32(got, 0);
		adduint32(got, 0);
	}
}

int
archreloc(Reloc *r, Sym *s, vlong *val)
{
	USED(s);
	switch(r->type) {
	case D_CONST:
		*val = r->add;
		return 0;
	case D_GOTOFF:
		*val = symaddr(r->sym) + r->add - symaddr(lookup(".got", 0));
		return 0;
	}
	return -1;
}

static void
addpltsym(Sym *s)
{
	Sym *plt, *got, *rel;
	
	if(s->plt >= 0)
		return;

	adddynsym(s);
	
	if(iself) {
		plt = lookup(".plt", 0);
		got = lookup(".got.plt", 0);
		rel = lookup(".rel.plt", 0);
		if(plt->size == 0)
			elfsetupplt();
		
		// jmpq *got+size
		adduint8(plt, 0xff);
		adduint8(plt, 0x25);
		addaddrplus(plt, got, got->size);
		
		// add to got: pointer to current pos in plt
		addaddrplus(got, plt, plt->size);
		
		// pushl $x
		adduint8(plt, 0x68);
		adduint32(plt, rel->size);
		
		// jmp .plt
		adduint8(plt, 0xe9);
		adduint32(plt, -(plt->size+4));
		
		// rel
		addaddrplus(rel, got, got->size-4);
		adduint32(rel, ELF32_R_INFO(s->dynid, R_386_JMP_SLOT));
		
		s->plt = plt->size - 16;
	} else if(HEADTYPE == Hdarwin) {
		// Same laziness as in 6l.
		
		Sym *plt;

		plt = lookup(".plt", 0);

		addgotsym(s);

		adduint32(lookup(".linkedit.plt", 0), s->dynid);

		// jmpq *got+size(IP)
		s->plt = plt->size;

		adduint8(plt, 0xff);
		adduint8(plt, 0x25);
		addaddrplus(plt, lookup(".got", 0), s->got);
	} else {
		diag("addpltsym: unsupported binary format");
	}
}

static void
addgotsym(Sym *s)
{
	Sym *got, *rel;
	
	if(s->got >= 0)
		return;
	
	adddynsym(s);
	got = lookup(".got", 0);
	s->got = got->size;
	adduint32(got, 0);
	
	if(iself) {
		rel = lookup(".rel", 0);
		addaddrplus(rel, got, s->got);
		adduint32(rel, ELF32_R_INFO(s->dynid, R_386_GLOB_DAT));
	} else if(HEADTYPE == Hdarwin) {
		adduint32(lookup(".linkedit.got", 0), s->dynid);
	} else {
		diag("addgotsym: unsupported binary format");
	}
}

void
adddynsym(Sym *s)
{
	Sym *d, *str;
	int t;
	char *name;
	
	if(s->dynid >= 0)
		return;
	
	if(s->dynimpname == nil)
		diag("adddynsym: no dynamic name for %s", s->name);

	if(iself) {
		s->dynid = nelfsym++;
		
		d = lookup(".dynsym", 0);

		/* name */
		name = s->dynimpname;
		if(name == nil)
			name = s->name;
		adduint32(d, addstring(lookup(".dynstr", 0), name));
		
		/* value */
		if(s->type == SDYNIMPORT)
			adduint32(d, 0);
		else
			addaddr(d, s);
		
		/* size */
		adduint32(d, 0);
	
		/* type */
		t = STB_GLOBAL << 4;
		if(s->dynexport && s->type == STEXT)
			t |= STT_FUNC;
		else
			t |= STT_OBJECT;
		adduint8(d, t);
		adduint8(d, 0);
	
		/* shndx */
		if(!s->dynexport && s->dynimpname != nil)
			adduint16(d, SHN_UNDEF);
		else {
			switch(s->type) {
			default:
			case STEXT:
				t = 11;
				break;
			case SRODATA:
				t = 12;
				break;
			case SDATA:
				t = 13;
				break;
			case SBSS:
				t = 14;
				break;
			}
			adduint16(d, t);
		}
	} else if(HEADTYPE == Hdarwin) {
		// Mach-O symbol nlist32
		d = lookup(".dynsym", 0);
		name = s->dynimpname;
		if(name == nil)
			name = s->name;
		s->dynid = d->size/12;
		// darwin still puts _ prefixes on all C symbols
		str = lookup(".dynstr", 0);
		adduint32(d, str->size);
		adduint8(str, '_');
		addstring(str, name);
		adduint8(d, 0x01);	// type - N_EXT - external symbol
		adduint8(d, 0);	// section
		adduint16(d, 0);	// desc
		adduint32(d, 0);	// value
	} else if(HEADTYPE != Hwindows) {
		diag("adddynsym: unsupported binary format");
	}
}

void
adddynlib(char *lib)
{
	Sym *s;
	
	if(!needlib(lib))
		return;
	
	if(iself) {
		s = lookup(".dynstr", 0);
		if(s->size == 0)
			addstring(s, "");
		elfwritedynent(lookup(".dynamic", 0), DT_NEEDED, addstring(s, lib));
	} else if(HEADTYPE == Hdarwin) {
		machoadddynlib(lib);
	} else if(HEADTYPE != Hwindows) {
		diag("adddynlib: unsupported binary format");
	}
}

void
doelf(void)
{
	Sym *s, *shstrtab, *dynstr;

	if(!iself)
		return;

	/* predefine strings we need for section headers */
	shstrtab = lookup(".shstrtab", 0);
	shstrtab->type = SELFROSECT;
	shstrtab->reachable = 1;

	elfstr[ElfStrEmpty] = addstring(shstrtab, "");
	elfstr[ElfStrText] = addstring(shstrtab, ".text");
	elfstr[ElfStrData] = addstring(shstrtab, ".data");
	elfstr[ElfStrBss] = addstring(shstrtab, ".bss");
	if(HEADTYPE == Hnetbsd)
		elfstr[ElfStrNoteNetbsdIdent] = addstring(shstrtab, ".note.netbsd.ident");
	addstring(shstrtab, ".elfdata");
	addstring(shstrtab, ".rodata");
	addstring(shstrtab, ".gosymtab");
	addstring(shstrtab, ".gopclntab");
	if(!debug['s']) {
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
		elfstr[ElfStrRel] = addstring(shstrtab, ".rel");
		elfstr[ElfStrRelPlt] = addstring(shstrtab, ".rel.plt");
		elfstr[ElfStrPlt] = addstring(shstrtab, ".plt");
		elfstr[ElfStrGnuVersion] = addstring(shstrtab, ".gnu.version");
		elfstr[ElfStrGnuVersionR] = addstring(shstrtab, ".gnu.version_r");

		/* dynamic symbol table - first entry all zeros */
		s = lookup(".dynsym", 0);
		s->type = SELFROSECT;
		s->reachable = 1;
		s->size += ELF32SYMSIZE;

		/* dynamic string table */
		s = lookup(".dynstr", 0);
		s->reachable = 1;
		s->type = SELFROSECT;
		if(s->size == 0)
			addstring(s, "");
		dynstr = s;

		/* relocation table */
		s = lookup(".rel", 0);
		s->reachable = 1;
		s->type = SELFROSECT;

		/* global offset table */
		s = lookup(".got", 0);
		s->reachable = 1;
		s->type = SELFSECT; // writable
		
		/* hash */
		s = lookup(".hash", 0);
		s->reachable = 1;
		s->type = SELFROSECT;

		/* got.plt */
		s = lookup(".got.plt", 0);
		s->reachable = 1;
		s->type = SELFSECT; // writable
		
		s = lookup(".plt", 0);
		s->reachable = 1;
		s->type = SELFROSECT;

		s = lookup(".rel.plt", 0);
		s->reachable = 1;
		s->type = SELFROSECT;
		
		s = lookup(".gnu.version", 0);
		s->reachable = 1;
		s->type = SELFROSECT;
		
		s = lookup(".gnu.version_r", 0);
		s->reachable = 1;
		s->type = SELFROSECT;

		elfsetupplt();

		/* define dynamic elf table */
		s = lookup(".dynamic", 0);
		s->reachable = 1;
		s->type = SELFSECT; // writable

		/*
		 * .dynamic table
		 */
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
		elfwritedynentsym(s, DT_PLTGOT, lookup(".got.plt", 0));
		elfwritedynent(s, DT_PLTREL, DT_REL);
		elfwritedynentsymsize(s, DT_PLTRELSZ, lookup(".rel.plt", 0));
		elfwritedynentsym(s, DT_JMPREL, lookup(".rel.plt", 0));

		elfwritedynent(s, DT_DEBUG, 0);

		// Do not write DT_NULL.  elfdynhash will finish it.
	}
}

void
shsym(Elf64_Shdr *sh, Sym *s)
{
	vlong addr;
	addr = symaddr(s);
	if(sh->flags&SHF_ALLOC)
		sh->addr = addr;
	sh->off = datoff(addr);
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
	int32 v, magic;
	int a, dynsym;
	uint32 symo, startva, dwarfoff, machlink, resoff;
	ElfEhdr *eh;
	ElfPhdr *ph, *pph;
	ElfShdr *sh;
	Section *sect;
	Sym *sym;
	int o;
	int i;

	if(debug['v'])
		Bprint(&bso, "%5.2f asmb\n", cputime());
	Bflush(&bso);

	sect = segtext.sect;
	cseek(sect->vaddr - segtext.vaddr + segtext.fileoff);
	codeblk(sect->vaddr, sect->len);

	/* output read-only data in text segment (rodata, gosymtab and pclntab) */
	for(sect = sect->next; sect != nil; sect = sect->next) {
		cseek(sect->vaddr - segtext.vaddr + segtext.fileoff);
		datblk(sect->vaddr, sect->len);
	}

	if(debug['v'])
		Bprint(&bso, "%5.2f datblk\n", cputime());
	Bflush(&bso);

	cseek(segdata.fileoff);
	datblk(segdata.vaddr, segdata.filelen);

	machlink = 0;
	if(HEADTYPE == Hdarwin) {
		if(debug['v'])
			Bprint(&bso, "%5.2f dwarf\n", cputime());

		dwarfoff = rnd(HEADR+segtext.len, INITRND) + rnd(segdata.filelen, INITRND);
		cseek(dwarfoff);

		segdwarf.fileoff = cpos();
		dwarfemitdebugsections();
		segdwarf.filelen = cpos() - segdwarf.fileoff;

		machlink = domacholink();
	}

	if(iself) {
		/* index of elf text section; needed by asmelfsym, double-checked below */
		/* !debug['d'] causes extra sections before the .text section */
		elftextsh = 2;
		if(!debug['d']) {
			elftextsh += 10;
			if(elfverneed)
				elftextsh += 2;
		}
		if(HEADTYPE == Hnetbsd)
			elftextsh += 1;
	}

	symsize = 0;
	spsize = 0;
	lcsize = 0;
	symo = 0;
	if(!debug['s']) {
		// TODO: rationalize
		if(debug['v'])
			Bprint(&bso, "%5.2f sym\n", cputime());
		Bflush(&bso);
		switch(HEADTYPE) {
		default:
			if(iself)
				goto Elfsym;
		case Hgarbunix:
			symo = rnd(HEADR+segtext.filelen, 8192)+segdata.filelen;
			break;
		case Hunixcoff:
			symo = rnd(HEADR+segtext.filelen, INITRND)+segdata.filelen;
			break;
		case Hplan9x32:
			symo = HEADR+segtext.filelen+segdata.filelen;
			break;
		case Hmsdoscom:
		case Hmsdosexe:
			debug['s'] = 1;
			symo = HEADR+segtext.filelen+segdata.filelen;
			break;
		case Hdarwin:
			symo = rnd(HEADR+segtext.filelen, INITRND)+rnd(segdata.filelen, INITRND)+machlink;
			break;
		Elfsym:
			symo = rnd(HEADR+segtext.filelen, INITRND)+segdata.filelen;
			symo = rnd(symo, INITRND);
			break;
		case Hwindows:
			symo = rnd(HEADR+segtext.filelen, PEFILEALIGN)+segdata.filelen;
			symo = rnd(symo, PEFILEALIGN);
			break;
		}
		cseek(symo);
		switch(HEADTYPE) {
		default:
			if(iself) {
				if(debug['v'])
					Bprint(&bso, "%5.2f elfsym\n", cputime());
				asmelfsym();
				cflush();
				cwrite(elfstrdat, elfstrsize);

				if(debug['v'])
					Bprint(&bso, "%5.2f dwarf\n", cputime());
				dwarfemitdebugsections();
			}
			break;
		case Hplan9x32:
			asmplan9sym();
			cflush();

			sym = lookup("pclntab", 0);
			if(sym != nil) {
				lcsize = sym->np;
				for(i=0; i < lcsize; i++)
					cput(sym->p[i]);
				
				cflush();
			}
			break;
		case Hwindows:
			if(debug['v'])
				Bprint(&bso, "%5.2f dwarf\n", cputime());
			dwarfemitdebugsections();
			break;
		}
	}
	if(debug['v'])
		Bprint(&bso, "%5.2f headr\n", cputime());
	Bflush(&bso);
	cseek(0L);
	switch(HEADTYPE) {
	default:
		if(iself)
			goto Elfput;
	case Hgarbunix:	/* garbage */
		lputb(0x160L<<16);		/* magic and sections */
		lputb(0L);			/* time and date */
		lputb(rnd(HEADR+segtext.filelen, 4096)+segdata.filelen);
		lputb(symsize);			/* nsyms */
		lputb((0x38L<<16)|7L);		/* size of optional hdr and flags */
		lputb((0413<<16)|0437L);		/* magic and version */
		lputb(rnd(HEADR+segtext.filelen, 4096));	/* sizes */
		lputb(segdata.filelen);
		lputb(segdata.len - segdata.filelen);
		lputb(entryvalue());		/* va of entry */
		lputb(INITTEXT-HEADR);		/* va of base of text */
		lputb(segdata.vaddr);			/* va of base of data */
		lputb(segdata.vaddr+segdata.filelen);		/* va of base of bss */
		lputb(~0L);			/* gp reg mask */
		lputb(0L);
		lputb(0L);
		lputb(0L);
		lputb(0L);
		lputb(~0L);			/* gp value ?? */
		break;
	case Hunixcoff:	/* unix coff */
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
		lputl(rnd(segtext.filelen, INITRND));	/* text sizes */
		lputl(segdata.filelen);			/* data sizes */
		lputl(segdata.len - segdata.filelen);			/* bss sizes */
		lputb(entryvalue());		/* va of entry */
		lputl(INITTEXT);		/* text start */
		lputl(segdata.vaddr);			/* data start */
		/*
		 * text section header
		 */
		s8put(".text");
		lputl(HEADR);			/* pa */
		lputl(HEADR);			/* va */
		lputl(segtext.filelen);		/* text size */
		lputl(HEADR);			/* file offset */
		lputl(0);			/* relocation */
		lputl(0);			/* line numbers */
		lputl(0);			/* relocation, line numbers */
		lputl(0x20);			/* flags text only */
		/*
		 * data section header
		 */
		s8put(".data");
		lputl(segdata.vaddr);			/* pa */
		lputl(segdata.vaddr);			/* va */
		lputl(segdata.filelen);			/* data size */
		lputl(HEADR+segtext.filelen);		/* file offset */
		lputl(0);			/* relocation */
		lputl(0);			/* line numbers */
		lputl(0);			/* relocation, line numbers */
		lputl(0x40);			/* flags data only */
		/*
		 * bss section header
		 */
		s8put(".bss");
		lputl(segdata.vaddr+segdata.filelen);		/* pa */
		lputl(segdata.vaddr+segdata.filelen);		/* va */
		lputl(segdata.len - segdata.filelen);			/* bss size */
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
		lputl(HEADR+segtext.filelen+segdata.filelen);	/* file offset */
		lputl(HEADR+segtext.filelen+segdata.filelen);	/* offset of syms */
		lputl(HEADR+segtext.filelen+segdata.filelen+symsize);/* offset of line numbers */
		lputl(0);			/* relocation, line numbers */
		lputl(0x200);			/* flags comment only */
		break;
	case Hplan9x32:	/* plan9 */
		magic = 4*11*11+7;
		lputb(magic);		/* magic */
		lputb(segtext.filelen);			/* sizes */
		lputb(segdata.filelen);
		lputb(segdata.len - segdata.filelen);
		lputb(symsize);			/* nsyms */
		lputb(entryvalue());		/* va of entry */
		lputb(spsize);			/* sp offsets */
		lputb(lcsize);			/* line offsets */
		break;
	case Hmsdoscom:
		/* MS-DOS .COM */
		break;
	case Hmsdosexe:
		/* fake MS-DOS .EXE */
		v = rnd(HEADR+segtext.filelen, INITRND)+segdata.filelen;
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

	case Hdarwin:
		asmbmacho();
		break;

	Elfput:
		eh = getElfEhdr();
		startva = INITTEXT - HEADR;
		resoff = ELFRESERVE;

		/* This null SHdr must appear before all others */
		newElfShdr(elfstr[ElfStrEmpty]);

		/* program header info */
		pph = newElfPhdr();
		pph->type = PT_PHDR;
		pph->flags = PF_R + PF_X;
		pph->off = eh->ehsize;
		pph->vaddr = INITTEXT - HEADR + pph->off;
		pph->paddr = INITTEXT - HEADR + pph->off;
		pph->align = INITRND;

		/*
		 * PHDR must be in a loaded segment. Adjust the text
		 * segment boundaries downwards to include it.
		 */
		o = segtext.vaddr - pph->vaddr;
		segtext.vaddr -= o;
		segtext.len += o;
		o = segtext.fileoff - pph->off;
		segtext.fileoff -= o;
		segtext.filelen += o;

		if(!debug['d']) {
			/* interpreter */
			sh = newElfShdr(elfstr[ElfStrInterp]);
			sh->type = SHT_PROGBITS;
			sh->flags = SHF_ALLOC;
			sh->addralign = 1;
			if(interpreter == nil) {
				switch(HEADTYPE) {
				case Hlinux:
					interpreter = linuxdynld;
					break;
				case Hfreebsd:
					interpreter = freebsddynld;
					break;
				case Hnetbsd:
					interpreter = netbsddynld;
					break;
				case Hopenbsd:
					interpreter = openbsddynld;
					break;
				}
			}
			resoff -= elfinterp(sh, startva, resoff, interpreter);

			ph = newElfPhdr();
			ph->type = PT_INTERP;
			ph->flags = PF_R;
			phsh(ph, sh);
		}

		if(HEADTYPE == Hnetbsd) {
			sh = newElfShdr(elfstr[ElfStrNoteNetbsdIdent]);
			sh->type = SHT_NOTE;
			sh->flags = SHF_ALLOC;
			sh->addralign = 4;
			resoff -= elfnetbsdsig(sh, startva, resoff);

			ph = newElfPhdr();
			ph->type = PT_NOTE;
			ph->flags = PF_R;
			phsh(ph, sh);
		}

		elfphload(&segtext);
		elfphload(&segdata);

		/* Dynamic linking sections */
		if(!debug['d']) {	/* -d suppresses dynamic loader format */
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
			
			if(elfverneed) {
				sh = newElfShdr(elfstr[ElfStrGnuVersion]);
				sh->type = SHT_GNU_VERSYM;
				sh->flags = SHF_ALLOC;
				sh->addralign = 2;
				sh->link = dynsym;
				sh->entsize = 2;
				shsym(sh, lookup(".gnu.version", 0));

				sh = newElfShdr(elfstr[ElfStrGnuVersionR]);
				sh->type = SHT_GNU_VERNEED;
				sh->flags = SHF_ALLOC;
				sh->addralign = 4;
				sh->info = elfverneed;
				sh->link = dynsym+1;  // dynstr
				shsym(sh, lookup(".gnu.version_r", 0));
			}

			sh = newElfShdr(elfstr[ElfStrRelPlt]);
			sh->type = SHT_REL;
			sh->flags = SHF_ALLOC;
			sh->entsize = ELF32RELSIZE;
			sh->addralign = 4;
			sh->link = dynsym;
			sh->info = eh->shnum;	// .plt
			shsym(sh, lookup(".rel.plt", 0));
			
			sh = newElfShdr(elfstr[ElfStrPlt]);
			sh->type = SHT_PROGBITS;
			sh->flags = SHF_ALLOC+SHF_EXECINSTR;
			sh->entsize = 4;
			sh->addralign = 4;
			shsym(sh, lookup(".plt", 0));

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

		sh = newElfShstrtab(elfstr[ElfStrShstrtab]);
		sh->type = SHT_STRTAB;
		sh->addralign = 1;
		shsym(sh, lookup(".shstrtab", 0));

		if(elftextsh != eh->shnum)
			diag("elftextsh = %d, want %d", elftextsh, eh->shnum);
		for(sect=segtext.sect; sect!=nil; sect=sect->next)
			elfshbits(sect);
		for(sect=segdata.sect; sect!=nil; sect=sect->next)
			elfshbits(sect);

		if(!debug['s']) {
			sh = newElfShdr(elfstr[ElfStrSymtab]);
			sh->type = SHT_SYMTAB;
			sh->off = symo;
			sh->size = symsize;
			sh->addralign = 4;
			sh->entsize = 16;
			sh->link = eh->shnum;	// link to strtab

			sh = newElfShdr(elfstr[ElfStrStrtab]);
			sh->type = SHT_STRTAB;
			sh->off = symo+symsize;
			sh->size = elfstrsize;
			sh->addralign = 1;

			dwarfaddelfheaders();
		}

		/* Main header */
		eh->ident[EI_MAG0] = '\177';
		eh->ident[EI_MAG1] = 'E';
		eh->ident[EI_MAG2] = 'L';
		eh->ident[EI_MAG3] = 'F';
		eh->ident[EI_CLASS] = ELFCLASS32;
		eh->ident[EI_DATA] = ELFDATA2LSB;
		eh->ident[EI_VERSION] = EV_CURRENT;
		switch(HEADTYPE) {
		case Hfreebsd:
			eh->ident[EI_OSABI] = ELFOSABI_FREEBSD;
			break;
		case Hnetbsd:
			eh->ident[EI_OSABI] = ELFOSABI_NETBSD;
			break;
		case Hopenbsd:
			eh->ident[EI_OSABI] = ELFOSABI_OPENBSD;
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

		cseek(0);
		a = 0;
		a += elfwritehdr();
		a += elfwritephdrs();
		a += elfwriteshdrs();
		a += elfwriteinterp(elfstr[ElfStrInterp]);
		if(HEADTYPE == Hnetbsd)
			a += elfwritenetbsdsig(elfstr[ElfStrNoteNetbsdIdent]);
		if(a > ELFRESERVE)	
			diag("ELFRESERVE too small: %d > %d", a, ELFRESERVE);
		break;

	case Hwindows:
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
genasmsym(void (*put)(Sym*, char*, int, vlong, vlong, int, Sym*))
{
	Auto *a;
	Sym *s;
	int h;

	s = lookup("etext", 0);
	if(s->type == STEXT)
		put(s, s->name, 'T', s->value, s->size, s->version, 0);

	for(h=0; h<NHASH; h++) {
		for(s=hash[h]; s!=S; s=s->hash) {
			if(s->hide)
				continue;
			switch(s->type&~SSUB) {
			case SCONST:
			case SRODATA:
			case SDATA:
			case SELFROSECT:
			case SMACHO:
			case SMACHOGOT:
			case STYPE:
			case SSTRING:
			case SGOSTRING:
			case SWINDOWS:
				if(!s->reachable)
					continue;
				put(s, s->name, 'D', symaddr(s), s->size, s->version, s->gotype);
				continue;

			case SBSS:
				if(!s->reachable)
					continue;
				put(s, s->name, 'B', symaddr(s), s->size, s->version, s->gotype);
				continue;

			case SFILE:
				put(nil, s->name, 'f', s->value, 0, s->version, 0);
				continue;
			}
		}
	}

	for(s = textp; s != nil; s = s->next) {
		if(s->text == nil)
			continue;

		/* filenames first */
		for(a=s->autom; a; a=a->link)
			if(a->type == D_FILE)
				put(nil, a->asym->name, 'z', a->aoffset, 0, 0, 0);
			else
			if(a->type == D_FILE1)
				put(nil, a->asym->name, 'Z', a->aoffset, 0, 0, 0);

		put(s, s->name, 'T', s->value, s->size, s->version, s->gotype);

		/* frame, auto and param after */
		put(nil, ".frame", 'm', s->text->to.offset+4, 0, 0, 0);

		for(a=s->autom; a; a=a->link)
			if(a->type == D_AUTO)
				put(nil, a->asym->name, 'a', -a->aoffset, 0, 0, a->gotype);
			else
			if(a->type == D_PARAM)
				put(nil, a->asym->name, 'p', a->aoffset, 0, 0, a->gotype);
	}
	if(debug['v'] || debug['n'])
		Bprint(&bso, "symsize = %d\n", symsize);
	Bflush(&bso);
}
