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
#include	"../ld/pe.h"

#define PADDR(a)	((uint32)(a) & ~0x80000000)

char linuxdynld[] = "/lib64/ld-linux-x86-64.so.2";
char freebsddynld[] = "/libexec/ld-elf.so.1";
char openbsddynld[] = "/usr/libexec/ld.so";
char netbsddynld[] = "/libexec/ld.elf_so";

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
	ElfStrShstrtab,
	ElfStrSymtab,
	ElfStrStrtab,
	ElfStrRelaPlt,
	ElfStrPlt,
	ElfStrGnuVersion,
	ElfStrGnuVersionR,
	ElfStrNoteNetbsdIdent,
	ElfStrNoteOpenbsdIdent,
	ElfStrNoPtrData,
	ElfStrNoPtrBss,
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
	p = smprint(".elfload.%s", name);
	s = lookup(p, 0);
	free(p);
	if(s->type == 0) {
		s->type = 100;	// avoid SDATA, etc.
		return 1;
	}
	return 0;
}

int nelfsym = 1;

static void addpltsym(Sym*);
static void addgotsym(Sym*);

void
adddynrel(Sym *s, Reloc *r)
{
	Sym *targ, *rela, *got;
	
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
	case 256 + R_X86_64_PC32:
		if(targ->dynimpname != nil && !targ->dynexport)
			diag("unexpected R_X86_64_PC32 relocation for dynamic symbol %s", targ->name);
		if(targ->type == 0 || targ->type == SXREF)
			diag("unknown symbol %s in pcrel", targ->name);
		r->type = D_PCREL;
		r->add += 4;
		return;
	
	case 256 + R_X86_64_PLT32:
		r->type = D_PCREL;
		r->add += 4;
		if(targ->dynimpname != nil && !targ->dynexport) {
			addpltsym(targ);
			r->sym = lookup(".plt", 0);
			r->add += targ->plt;
		}
		return;
	
	case 256 + R_X86_64_GOTPCREL:
		if(targ->dynimpname == nil || targ->dynexport) {
			// have symbol
			if(r->off >= 2 && s->p[r->off-2] == 0x8b) {
				// turn MOVQ of GOT entry into LEAQ of symbol itself
				s->p[r->off-2] = 0x8d;
				r->type = D_PCREL;
				r->add += 4;
				return;
			}
			// fall back to using GOT and hope for the best (CMOV*)
			// TODO: just needs relocation, no need to put in .dynsym
			targ->dynimpname = targ->name;
		}
		addgotsym(targ);
		r->type = D_PCREL;
		r->sym = lookup(".got", 0);
		r->add += 4;
		r->add += targ->got;
		return;
	
	case 256 + R_X86_64_64:
		if(targ->dynimpname != nil && !targ->dynexport)
			diag("unexpected R_X86_64_64 relocation for dynamic symbol %s", targ->name);
		r->type = D_ADDR;
		return;
	
	// Handle relocations found in Mach-O object files.
	case 512 + MACHO_X86_64_RELOC_UNSIGNED*2 + 0:
	case 512 + MACHO_X86_64_RELOC_SIGNED*2 + 0:
	case 512 + MACHO_X86_64_RELOC_BRANCH*2 + 0:
		// TODO: What is the difference between all these?
		r->type = D_ADDR;
		if(targ->dynimpname != nil && !targ->dynexport)
			diag("unexpected reloc for dynamic symbol %s", targ->name);
		return;

	case 512 + MACHO_X86_64_RELOC_BRANCH*2 + 1:
		if(targ->dynimpname != nil && !targ->dynexport) {
			addpltsym(targ);
			r->sym = lookup(".plt", 0);
			r->add = targ->plt;
			r->type = D_PCREL;
			return;
		}
		// fall through
	case 512 + MACHO_X86_64_RELOC_UNSIGNED*2 + 1:
	case 512 + MACHO_X86_64_RELOC_SIGNED*2 + 1:
	case 512 + MACHO_X86_64_RELOC_SIGNED_1*2 + 1:
	case 512 + MACHO_X86_64_RELOC_SIGNED_2*2 + 1:
	case 512 + MACHO_X86_64_RELOC_SIGNED_4*2 + 1:
		r->type = D_PCREL;
		if(targ->dynimpname != nil && !targ->dynexport)
			diag("unexpected pc-relative reloc for dynamic symbol %s", targ->name);
		return;

	case 512 + MACHO_X86_64_RELOC_GOT_LOAD*2 + 1:
		if(targ->dynimpname == nil || targ->dynexport) {
			// have symbol
			// turn MOVQ of GOT entry into LEAQ of symbol itself
			if(r->off < 2 || s->p[r->off-2] != 0x8b) {
				diag("unexpected GOT_LOAD reloc for non-dynamic symbol %s", targ->name);
				return;
			}
			s->p[r->off-2] = 0x8d;
			r->type = D_PCREL;
			return;
		}
		// fall through
	case 512 + MACHO_X86_64_RELOC_GOT*2 + 1:
		if(targ->dynimpname == nil || targ->dynexport)
			diag("unexpected GOT reloc for non-dynamic symbol %s", targ->name);
		addgotsym(targ);
		r->type = D_PCREL;
		r->sym = lookup(".got", 0);
		r->add += targ->got;
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
			rela = lookup(".rela", 0);
			addaddrplus(rela, s, r->off);
			if(r->siz == 8)
				adduint64(rela, ELF64_R_INFO(targ->dynid, R_X86_64_64));
			else
				adduint64(rela, ELF64_R_INFO(targ->dynid, R_X86_64_32));
			adduint64(rela, r->add);
			r->type = 256;	// ignore during relocsym
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
			adduint64(got, 0);
			adduint32(lookup(".linkedit.got", 0), targ->dynid);
			r->type = 256;	// ignore during relocsym
			return;
		}
		break;
	}
	
	cursym = s;
	diag("unsupported relocation for dynamic symbol %s (type=%d stype=%d)", targ->name, r->type, targ->type);
}

int
archreloc(Reloc *r, Sym *s, vlong *val)
{
	USED(r);
	USED(s);
	USED(val);
	return -1;
}

static void
elfsetupplt(void)
{
	Sym *plt, *got;

	plt = lookup(".plt", 0);
	got = lookup(".got.plt", 0);
	if(plt->size == 0) {
		// pushq got+8(IP)
		adduint8(plt, 0xff);
		adduint8(plt, 0x35);
		addpcrelplus(plt, got, 8);
		
		// jmpq got+16(IP)
		adduint8(plt, 0xff);
		adduint8(plt, 0x25);
		addpcrelplus(plt, got, 16);
		
		// nopl 0(AX)
		adduint32(plt, 0x00401f0f);
		
		// assume got->size == 0 too
		addaddrplus(got, lookup(".dynamic", 0), 0);
		adduint64(got, 0);
		adduint64(got, 0);
	}
}

static void
addpltsym(Sym *s)
{
	if(s->plt >= 0)
		return;
	
	adddynsym(s);
	
	if(iself) {
		Sym *plt, *got, *rela;

		plt = lookup(".plt", 0);
		got = lookup(".got.plt", 0);
		rela = lookup(".rela.plt", 0);
		if(plt->size == 0)
			elfsetupplt();
		
		// jmpq *got+size(IP)
		adduint8(plt, 0xff);
		adduint8(plt, 0x25);
		addpcrelplus(plt, got, got->size);
	
		// add to got: pointer to current pos in plt
		addaddrplus(got, plt, plt->size);
		
		// pushq $x
		adduint8(plt, 0x68);
		adduint32(plt, (got->size-24-8)/8);
		
		// jmpq .plt
		adduint8(plt, 0xe9);
		adduint32(plt, -(plt->size+4));
		
		// rela
		addaddrplus(rela, got, got->size-8);
		adduint64(rela, ELF64_R_INFO(s->dynid, R_X86_64_JMP_SLOT));
		adduint64(rela, 0);
		
		s->plt = plt->size - 16;
	} else if(HEADTYPE == Hdarwin) {
		// To do lazy symbol lookup right, we're supposed
		// to tell the dynamic loader which library each 
		// symbol comes from and format the link info
		// section just so.  I'm too lazy (ha!) to do that
		// so for now we'll just use non-lazy pointers,
		// which don't need to be told which library to use.
		//
		// http://networkpx.blogspot.com/2009/09/about-lcdyldinfoonly-command.html
		// has details about what we're avoiding.

		Sym *plt;
		
		addgotsym(s);
		plt = lookup(".plt", 0);

		adduint32(lookup(".linkedit.plt", 0), s->dynid);

		// jmpq *got+size(IP)
		s->plt = plt->size;

		adduint8(plt, 0xff);
		adduint8(plt, 0x25);
		addpcrelplus(plt, lookup(".got", 0), s->got);
	} else {
		diag("addpltsym: unsupported binary format");
	}
}

static void
addgotsym(Sym *s)
{
	Sym *got, *rela;

	if(s->got >= 0)
		return;

	adddynsym(s);
	got = lookup(".got", 0);
	s->got = got->size;
	adduint64(got, 0);

	if(iself) {
		rela = lookup(".rela", 0);
		addaddrplus(rela, got, s->got);
		adduint64(rela, ELF64_R_INFO(s->dynid, R_X86_64_GLOB_DAT));
		adduint64(rela, 0);
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

		name = s->dynimpname;
		if(name == nil)
			name = s->name;
		adduint32(d, addstring(lookup(".dynstr", 0), name));
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
	
		/* value */
		if(s->type == SDYNIMPORT)
			adduint64(d, 0);
		else
			addaddr(d, s);
	
		/* size of object */
		adduint64(d, 0);
	
		if(!s->dynexport && s->dynimplib && needlib(s->dynimplib)) {
			elfwritedynent(lookup(".dynamic", 0), DT_NEEDED,
				addstring(lookup(".dynstr", 0), s->dynimplib));
		}
	} else if(HEADTYPE == Hdarwin) {
		// Mach-o symbol nlist64
		d = lookup(".dynsym", 0);
		name = s->dynimpname;
		if(name == nil)
			name = s->name;
		s->dynid = d->size/16;
		// darwin still puts _ prefixes on all C symbols
		str = lookup(".dynstr", 0);
		adduint32(d, str->size);
		adduint8(str, '_');
		addstring(str, name);
		if(s->type == SDYNIMPORT) {
			adduint8(d, 0x01);	// type - N_EXT - external symbol
			adduint8(d, 0);	// section
		} else {
			adduint8(d, 0x0f);
			switch(s->type) {
			default:
			case STEXT:
				adduint8(d, 1);
				break;
			case SDATA:
				adduint8(d, 2);
				break;
			case SBSS:
				adduint8(d, 4);
				break;
			}
		}
		adduint16(d, 0);	// desc
		if(s->type == SDYNIMPORT)
			adduint64(d, 0);	// value
		else
			addaddr(d, s);
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
	} else {
		diag("adddynlib: unsupported binary format");
	}
}

void
doelf(void)
{
	Sym *s, *shstrtab, *dynstr;

	if(HEADTYPE != Hlinux && HEADTYPE != Hfreebsd && HEADTYPE != Hopenbsd && HEADTYPE != Hnetbsd)
		return;

	/* predefine strings we need for section headers */
	shstrtab = lookup(".shstrtab", 0);
	shstrtab->type = SELFROSECT;
	shstrtab->reachable = 1;

	elfstr[ElfStrEmpty] = addstring(shstrtab, "");
	elfstr[ElfStrText] = addstring(shstrtab, ".text");
	elfstr[ElfStrNoPtrData] = addstring(shstrtab, ".noptrdata");
	elfstr[ElfStrData] = addstring(shstrtab, ".data");
	elfstr[ElfStrBss] = addstring(shstrtab, ".bss");
	elfstr[ElfStrNoPtrBss] = addstring(shstrtab, ".noptrbss");
	if(HEADTYPE == Hnetbsd)
		elfstr[ElfStrNoteNetbsdIdent] = addstring(shstrtab, ".note.netbsd.ident");
	if(HEADTYPE == Hopenbsd)
		elfstr[ElfStrNoteOpenbsdIdent] = addstring(shstrtab, ".note.openbsd.ident");
	addstring(shstrtab, ".elfdata");
	addstring(shstrtab, ".rodata");
	addstring(shstrtab, ".gcdata");
	addstring(shstrtab, ".gcbss");
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
		elfstr[ElfStrRela] = addstring(shstrtab, ".rela");
		elfstr[ElfStrRelaPlt] = addstring(shstrtab, ".rela.plt");
		elfstr[ElfStrPlt] = addstring(shstrtab, ".plt");
		elfstr[ElfStrGnuVersion] = addstring(shstrtab, ".gnu.version");
		elfstr[ElfStrGnuVersionR] = addstring(shstrtab, ".gnu.version_r");

		/* dynamic symbol table - first entry all zeros */
		s = lookup(".dynsym", 0);
		s->type = SELFROSECT;
		s->reachable = 1;
		s->size += ELF64SYMSIZE;

		/* dynamic string table */
		s = lookup(".dynstr", 0);
		s->type = SELFROSECT;
		s->reachable = 1;
		if(s->size == 0)
			addstring(s, "");
		dynstr = s;

		/* relocation table */
		s = lookup(".rela", 0);
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

		s = lookup(".got.plt", 0);
		s->reachable = 1;
		s->type = SELFSECT; // writable

		s = lookup(".plt", 0);
		s->reachable = 1;
		s->type = SELFROSECT;
		
		elfsetupplt();
		
		s = lookup(".rela.plt", 0);
		s->reachable = 1;
		s->type = SELFROSECT;
		
		s = lookup(".gnu.version", 0);
		s->reachable = 1;
		s->type = SELFROSECT;
		
		s = lookup(".gnu.version_r", 0);
		s->reachable = 1;
		s->type = SELFROSECT;

		/* define dynamic elf table */
		s = lookup(".dynamic", 0);
		s->reachable = 1;
		s->type = SELFSECT; // writable

		/*
		 * .dynamic table
		 */
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
		
		elfwritedynentsym(s, DT_PLTGOT, lookup(".got.plt", 0));
		elfwritedynent(s, DT_PLTREL, DT_RELA);
		elfwritedynentsymsize(s, DT_PLTRELSZ, lookup(".rela.plt", 0));
		elfwritedynentsym(s, DT_JMPREL, lookup(".rela.plt", 0));
		
		elfwritedynent(s, DT_DEBUG, 0);

		// Do not write DT_NULL.  elfdynhash will finish it.
	}
}

void
shsym(ElfShdr *sh, Sym *s)
{
	vlong addr;
	addr = symaddr(s);
	if(sh->flags&SHF_ALLOC)
		sh->addr = addr;
	sh->off = datoff(addr);
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
	int32 magic;
	int a, dynsym;
	vlong vl, startva, symo, dwarfoff, machlink, resoff;
	ElfEhdr *eh;
	ElfPhdr *ph, *pph;
	ElfShdr *sh;
	Section *sect;
	Sym *sym;
	int i, o;

	if(debug['v'])
		Bprint(&bso, "%5.2f asmb\n", cputime());
	Bflush(&bso);

	elftextsh = 0;
	
	if(debug['v'])
		Bprint(&bso, "%5.2f codeblk\n", cputime());
	Bflush(&bso);

	sect = segtext.sect;
	cseek(sect->vaddr - segtext.vaddr + segtext.fileoff);
	codeblk(sect->vaddr, sect->len);

	/* output read-only data in text segment (rodata, gosymtab, pclntab, ...) */
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

	switch(HEADTYPE) {
	default:
		diag("unknown header type %d", HEADTYPE);
	case Hplan9x32:
	case Hplan9x64:
	case Helf:
		break;
	case Hdarwin:
		debug['8'] = 1;	/* 64-bit addresses */
		break;
	case Hlinux:
	case Hfreebsd:
	case Hnetbsd:
	case Hopenbsd:
		debug['8'] = 1;	/* 64-bit addresses */
		/* index of elf text section; needed by asmelfsym, double-checked below */
		/* !debug['d'] causes extra sections before the .text section */
		elftextsh = 2;
		if(!debug['d']) {
			elftextsh += 10;
			if(elfverneed)
				elftextsh += 2;
		}
		if(HEADTYPE == Hnetbsd || HEADTYPE == Hopenbsd)
			elftextsh += 1;
		break;
	case Hwindows:
		break;
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
		case Hplan9x64:
		case Helf:
			debug['s'] = 1;
			symo = HEADR+segtext.len+segdata.filelen;
			break;
		case Hdarwin:
			symo = rnd(HEADR+segtext.len, INITRND)+rnd(segdata.filelen, INITRND)+machlink;
			break;
		case Hlinux:
		case Hfreebsd:
		case Hnetbsd:
		case Hopenbsd:
			symo = rnd(HEADR+segtext.len, INITRND)+segdata.filelen;
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
				cseek(symo);
				asmelfsym();
				cflush();
				cwrite(elfstrdat, elfstrsize);

				if(debug['v'])
				       Bprint(&bso, "%5.2f dwarf\n", cputime());

				dwarfemitdebugsections();
			}
			break;
		case Hplan9x64:
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
	case Hplan9x64:	/* plan9 */
		magic = 4*26*26+7;
		magic |= 0x00008000;		/* fat header */
		lputb(magic);			/* magic */
		lputb(segtext.filelen);			/* sizes */
		lputb(segdata.filelen);
		lputb(segdata.len - segdata.filelen);
		lputb(symsize);			/* nsyms */
		vl = entryvalue();
		lputb(PADDR(vl));		/* va of entry */
		lputb(spsize);			/* sp offsets */
		lputb(lcsize);			/* line offsets */
		vputb(vl);			/* va of entry */
		break;
	case Hplan9x32:	/* plan9 */
		magic = 4*26*26+7;
		lputb(magic);			/* magic */
		lputb(segtext.filelen);		/* sizes */
		lputb(segdata.filelen);
		lputb(segdata.len - segdata.filelen);
		lputb(symsize);			/* nsyms */
		lputb(entryvalue());		/* va of entry */
		lputb(spsize);			/* sp offsets */
		lputb(lcsize);			/* line offsets */
		break;
	case Hdarwin:
		asmbmacho();
		break;
	case Hlinux:
	case Hfreebsd:
	case Hnetbsd:
	case Hopenbsd:
		/* elf amd-64 */

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

		if(HEADTYPE == Hnetbsd || HEADTYPE == Hopenbsd) {
			sh = nil;
			switch(HEADTYPE) {
			case Hnetbsd:
				sh = newElfShdr(elfstr[ElfStrNoteNetbsdIdent]);
				resoff -= elfnetbsdsig(sh, startva, resoff);
				break;
			case Hopenbsd:
				sh = newElfShdr(elfstr[ElfStrNoteOpenbsdIdent]);
				resoff -= elfopenbsdsig(sh, startva, resoff);
				break;
			}

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
				sh->addralign = 8;
				sh->info = elfverneed;
				sh->link = dynsym+1;  // dynstr
				shsym(sh, lookup(".gnu.version_r", 0));
			}

			sh = newElfShdr(elfstr[ElfStrRelaPlt]);
			sh->type = SHT_RELA;
			sh->flags = SHF_ALLOC;
			sh->entsize = ELF64RELASIZE;
			sh->addralign = 8;
			sh->link = dynsym;
			sh->info = eh->shnum;	// .plt
			shsym(sh, lookup(".rela.plt", 0));

			sh = newElfShdr(elfstr[ElfStrPlt]);
			sh->type = SHT_PROGBITS;
			sh->flags = SHF_ALLOC+SHF_EXECINSTR;
			sh->entsize = 16;
			sh->addralign = 4;
			shsym(sh, lookup(".plt", 0));

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
		
		ph = newElfPhdr();
		ph->type = PT_PAX_FLAGS;
		ph->flags = 0x2a00; // mprotect, randexec, emutramp disabled
		ph->align = 8;

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
			sh->addralign = 8;
			sh->entsize = 24;
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
		if(HEADTYPE == Hfreebsd)
			eh->ident[EI_OSABI] = ELFOSABI_FREEBSD;
		else if(HEADTYPE == Hnetbsd)
			eh->ident[EI_OSABI] = ELFOSABI_NETBSD;
		else if(HEADTYPE == Hopenbsd)
			eh->ident[EI_OSABI] = ELFOSABI_OPENBSD;
		eh->ident[EI_CLASS] = ELFCLASS64;
		eh->ident[EI_DATA] = ELFDATA2LSB;
		eh->ident[EI_VERSION] = EV_CURRENT;

		eh->type = ET_EXEC;
		eh->machine = EM_X86_64;
		eh->version = EV_CURRENT;
		eh->entry = entryvalue();

		pph->filesz = eh->phnum * eh->phentsize;
		pph->memsz = pph->filesz;

		cseek(0);
		a = 0;
		a += elfwritehdr();
		a += elfwritephdrs();
		a += elfwriteshdrs();
		a += elfwriteinterp(elfstr[ElfStrInterp]);
		if(HEADTYPE == Hnetbsd)
			a += elfwritenetbsdsig(elfstr[ElfStrNoteNetbsdIdent]);
		if(HEADTYPE == Hopenbsd)
			a += elfwriteopenbsdsig(elfstr[ElfStrNoteOpenbsdIdent]);
		if(a > ELFRESERVE)	
			diag("ELFRESERVE too small: %d > %d", a, ELFRESERVE);
		break;
	case Hwindows:
		asmbpe();
		break;
	}
	cflush();
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
genasmsym(void (*put)(Sym*, char*, int, vlong, vlong, int, Sym*))
{
	Auto *a;
	Sym *s;

	s = lookup("etext", 0);
	if(s->type == STEXT)
		put(s, s->name, 'T', s->value, s->size, s->version, 0);

	for(s=allsym; s!=S; s=s->allsym) {
		if(s->hide)
			continue;
		switch(s->type&SMASK) {
		case SCONST:
		case SRODATA:
		case SSYMTAB:
		case SPCLNTAB:
		case SDATA:
		case SNOPTRDATA:
		case SELFROSECT:
		case SMACHOGOT:
		case STYPE:
		case SSTRING:
		case SGOSTRING:
		case SWINDOWS:
		case SGCDATA:
		case SGCBSS:
			if(!s->reachable)
				continue;
			put(s, s->name, 'D', symaddr(s), s->size, s->version, s->gotype);
			continue;

		case SBSS:
		case SNOPTRBSS:
			if(!s->reachable)
				continue;
			if(s->np > 0)
				diag("%s should not be bss (size=%d type=%d special=%d)", s->name, (int)s->np, s->type, s->special);
			put(s, s->name, 'B', symaddr(s), s->size, s->version, s->gotype);
			continue;

		case SFILE:
			put(nil, s->name, 'f', s->value, 0, s->version, 0);
			continue;
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
		put(nil, ".frame", 'm', s->text->to.offset+8, 0, 0, 0);

		for(a=s->autom; a; a=a->link)
			if(a->type == D_AUTO)
				put(nil, a->asym->name, 'a', -a->aoffset, 0, 0, a->gotype);
			else
			if(a->type == D_PARAM)
				put(nil, a->asym->name, 'p', a->aoffset, 0, 0, a->gotype);
	}
	if(debug['v'] || debug['n'])
		Bprint(&bso, "symsize = %ud\n", symsize);
	Bflush(&bso);
}
