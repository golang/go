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
	free(p);
	if(s->type == 0) {
		s->type = 100;	// avoid SDATA, etc.
		return 1;
	}
	return 0;
}

int	nelfsym = 1;

static void	addpltsym(Sym*);
static void	addgotsym(Sym*);

Sym *
lookuprel(void)
{
	return lookup(".rel", 0);
}

void
adddynrela(Sym *rela, Sym *s, Reloc *r)
{
	USED(rela);
	USED(s);
	USED(r);
	sysfatal("adddynrela not implemented");
}

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
		if(targ->type == SDYNIMPORT)
			diag("unexpected R_386_PC32 relocation for dynamic symbol %s", targ->name);
		if(targ->type == 0 || targ->type == SXREF)
			diag("unknown symbol %s in pcrel", targ->name);
		r->type = D_PCREL;
		r->add += 4;
		return;

	case 256 + R_386_PLT32:
		r->type = D_PCREL;
		r->add += 4;
		if(targ->type == SDYNIMPORT) {
			addpltsym(targ);
			r->sym = lookup(".plt", 0);
			r->add += targ->plt;
		}
		return;		
	
	case 256 + R_386_GOT32:
		if(targ->type != SDYNIMPORT) {
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
		if(targ->type == SDYNIMPORT)
			diag("unexpected R_386_32 relocation for dynamic symbol %s", targ->name);
		r->type = D_ADDR;
		return;
	
	case 512 + MACHO_GENERIC_RELOC_VANILLA*2 + 0:
		r->type = D_ADDR;
		if(targ->type == SDYNIMPORT)
			diag("unexpected reloc for dynamic symbol %s", targ->name);
		return;
	
	case 512 + MACHO_GENERIC_RELOC_VANILLA*2 + 1:
		if(targ->type == SDYNIMPORT) {
			addpltsym(targ);
			r->sym = lookup(".plt", 0);
			r->add = targ->plt;
			r->type = D_PCREL;
			return;
		}
		r->type = D_PCREL;
		return;
	
	case 512 + MACHO_FAKE_GOTPCREL:
		if(targ->type != SDYNIMPORT) {
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
	if(targ->type != SDYNIMPORT)
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

int
elfreloc1(Reloc *r, vlong sectoff)
{
	int32 elfsym;

	LPUT(sectoff);

	elfsym = r->xsym->elfsym;
	switch(r->type) {
	default:
		return -1;

	case D_ADDR:
		if(r->siz == 4)
			LPUT(R_386_32 | elfsym<<8);
		else
			return -1;
		break;

	case D_PCREL:
		if(r->siz == 4)
			LPUT(R_386_PC32 | elfsym<<8);
		else
			return -1;
		break;
	
	case D_TLS:
		if(r->siz == 4)
			LPUT(R_386_TLS_LE | elfsym<<8);
		else
			return -1;
	}

	return 0;
}

int
machoreloc1(Reloc *r, vlong sectoff)
{
	uint32 v;
	Sym *rs;
	
	rs = r->xsym;

	if(rs->type == SHOSTOBJ) {
		if(rs->dynid < 0) {
			diag("reloc %d to non-macho symbol %s type=%d", r->type, rs->name, rs->type);
			return -1;
		}
		v = rs->dynid;			
		v |= 1<<27; // external relocation
	} else {
		v = rs->sect->extnum;
		if(v == 0) {
			diag("reloc %d to symbol %s in non-macho section %s type=%d", r->type, rs->name, rs->sect->name, rs->type);
			return -1;
		}
	}

	switch(r->type) {
	default:
		return -1;
	case D_ADDR:
		v |= MACHO_GENERIC_RELOC_VANILLA<<28;
		break;
	case D_PCREL:
		v |= 1<<24; // pc-relative bit
		v |= MACHO_GENERIC_RELOC_VANILLA<<28;
		break;
	}
	
	switch(r->siz) {
	default:
		return -1;
	case 1:
		v |= 0<<25;
		break;
	case 2:
		v |= 1<<25;
		break;
	case 4:
		v |= 2<<25;
		break;
	case 8:
		v |= 3<<25;
		break;
	}

	LPUT(sectoff);
	LPUT(v);
	return 0;
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

void
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
	Sym *d;
	int t;
	char *name;
	
	if(s->dynid >= 0)
		return;
	
	if(iself) {
		s->dynid = nelfsym++;
		
		d = lookup(".dynsym", 0);

		/* name */
		name = s->extname;
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
		if(s->cgoexport && (s->type&SMASK) == STEXT)
			t |= STT_FUNC;
		else
			t |= STT_OBJECT;
		adduint8(d, t);
		adduint8(d, 0);
	
		/* shndx */
		if(s->type == SDYNIMPORT)
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
		diag("adddynsym: missed symbol %s (%s)", s->name, s->extname);
	} else if(HEADTYPE == Hwindows) {
		// already taken care of
	} else {
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
asmb(void)
{
	int32 v, magic;
	uint32 symo, dwarfoff, machlink;
	Section *sect;
	Sym *sym;
	int i;

	if(debug['v'])
		Bprint(&bso, "%5.2f asmb\n", cputime());
	Bflush(&bso);

	if(iself)
		asmbelfsetup();

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
				
				if(linkmode == LinkExternal)
					elfemitreloc();
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
		case Hdarwin:
			if(linkmode == LinkExternal)
				machoemitreloc();
			break;
		}
	}
	if(debug['v'])
		Bprint(&bso, "%5.2f headr\n", cputime());
	Bflush(&bso);
	cseek(0L);
	switch(HEADTYPE) {
	default:
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
	case Hlinux:
	case Hfreebsd:
	case Hnetbsd:
	case Hopenbsd:
		asmbelf(symo);
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
