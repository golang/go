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
char dragonflydynld[] = "/usr/libexec/ld-elf.so.2";
char solarisdynld[] = "/lib/amd64/ld.so.1";

char	zeroes[32];

static int
needlib(char *name)
{
	char *p;
	LSym *s;

	if(*name == '\0')
		return 0;

	/* reuse hash code in symbol table */
	p = smprint(".elfload.%s", name);
	s = linklookup(ctxt, p, 0);
	free(p);
	if(s->type == 0) {
		s->type = 100;	// avoid SDATA, etc.
		return 1;
	}
	return 0;
}

int nelfsym = 1;

static void addpltsym(LSym*);
static void addgotsym(LSym*);

void
adddynrela(LSym *rela, LSym *s, Reloc *r)
{
	addaddrplus(ctxt, rela, s, r->off);
	adduint64(ctxt, rela, R_X86_64_RELATIVE);
	addaddrplus(ctxt, rela, r->sym, r->add); // Addend
}

void
adddynrel(LSym *s, Reloc *r)
{
	LSym *targ, *rela, *got;
	
	targ = r->sym;
	ctxt->cursym = s;

	switch(r->type) {
	default:
		if(r->type >= 256) {
			diag("unexpected relocation type %d", r->type);
			return;
		}
		break;

	// Handle relocations found in ELF object files.
	case 256 + R_X86_64_PC32:
		if(targ->type == SDYNIMPORT)
			diag("unexpected R_X86_64_PC32 relocation for dynamic symbol %s", targ->name);
		if(targ->type == 0 || targ->type == SXREF)
			diag("unknown symbol %s in pcrel", targ->name);
		r->type = R_PCREL;
		r->add += 4;
		return;
	
	case 256 + R_X86_64_PLT32:
		r->type = R_PCREL;
		r->add += 4;
		if(targ->type == SDYNIMPORT) {
			addpltsym(targ);
			r->sym = linklookup(ctxt, ".plt", 0);
			r->add += targ->plt;
		}
		return;
	
	case 256 + R_X86_64_GOTPCREL:
		if(targ->type != SDYNIMPORT) {
			// have symbol
			if(r->off >= 2 && s->p[r->off-2] == 0x8b) {
				// turn MOVQ of GOT entry into LEAQ of symbol itself
				s->p[r->off-2] = 0x8d;
				r->type = R_PCREL;
				r->add += 4;
				return;
			}
			// fall back to using GOT and hope for the best (CMOV*)
			// TODO: just needs relocation, no need to put in .dynsym
		}
		addgotsym(targ);
		r->type = R_PCREL;
		r->sym = linklookup(ctxt, ".got", 0);
		r->add += 4;
		r->add += targ->got;
		return;
	
	case 256 + R_X86_64_64:
		if(targ->type == SDYNIMPORT)
			diag("unexpected R_X86_64_64 relocation for dynamic symbol %s", targ->name);
		r->type = R_ADDR;
		return;
	
	// Handle relocations found in Mach-O object files.
	case 512 + MACHO_X86_64_RELOC_UNSIGNED*2 + 0:
	case 512 + MACHO_X86_64_RELOC_SIGNED*2 + 0:
	case 512 + MACHO_X86_64_RELOC_BRANCH*2 + 0:
		// TODO: What is the difference between all these?
		r->type = R_ADDR;
		if(targ->type == SDYNIMPORT)
			diag("unexpected reloc for dynamic symbol %s", targ->name);
		return;

	case 512 + MACHO_X86_64_RELOC_BRANCH*2 + 1:
		if(targ->type == SDYNIMPORT) {
			addpltsym(targ);
			r->sym = linklookup(ctxt, ".plt", 0);
			r->add = targ->plt;
			r->type = R_PCREL;
			return;
		}
		// fall through
	case 512 + MACHO_X86_64_RELOC_UNSIGNED*2 + 1:
	case 512 + MACHO_X86_64_RELOC_SIGNED*2 + 1:
	case 512 + MACHO_X86_64_RELOC_SIGNED_1*2 + 1:
	case 512 + MACHO_X86_64_RELOC_SIGNED_2*2 + 1:
	case 512 + MACHO_X86_64_RELOC_SIGNED_4*2 + 1:
		r->type = R_PCREL;
		if(targ->type == SDYNIMPORT)
			diag("unexpected pc-relative reloc for dynamic symbol %s", targ->name);
		return;

	case 512 + MACHO_X86_64_RELOC_GOT_LOAD*2 + 1:
		if(targ->type != SDYNIMPORT) {
			// have symbol
			// turn MOVQ of GOT entry into LEAQ of symbol itself
			if(r->off < 2 || s->p[r->off-2] != 0x8b) {
				diag("unexpected GOT_LOAD reloc for non-dynamic symbol %s", targ->name);
				return;
			}
			s->p[r->off-2] = 0x8d;
			r->type = R_PCREL;
			return;
		}
		// fall through
	case 512 + MACHO_X86_64_RELOC_GOT*2 + 1:
		if(targ->type != SDYNIMPORT)
			diag("unexpected GOT reloc for non-dynamic symbol %s", targ->name);
		addgotsym(targ);
		r->type = R_PCREL;
		r->sym = linklookup(ctxt, ".got", 0);
		r->add += targ->got;
		return;
	}
	
	// Handle references to ELF symbols from our own object files.
	if(targ->type != SDYNIMPORT)
		return;

	switch(r->type) {
	case R_CALL:
	case R_PCREL:
		addpltsym(targ);
		r->sym = linklookup(ctxt, ".plt", 0);
		r->add = targ->plt;
		return;
	
	case R_ADDR:
		if(s->type == STEXT && iself) {
			// The code is asking for the address of an external
			// function.  We provide it with the address of the
			// correspondent GOT symbol.
			addgotsym(targ);
			r->sym = linklookup(ctxt, ".got", 0);
			r->add += targ->got;
			return;
		}
		if(s->type != SDATA)
			break;
		if(iself) {
			adddynsym(ctxt, targ);
			rela = linklookup(ctxt, ".rela", 0);
			addaddrplus(ctxt, rela, s, r->off);
			if(r->siz == 8)
				adduint64(ctxt, rela, ELF64_R_INFO(targ->dynid, R_X86_64_64));
			else
				adduint64(ctxt, rela, ELF64_R_INFO(targ->dynid, R_X86_64_32));
			adduint64(ctxt, rela, r->add);
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
			adddynsym(ctxt, targ);
			got = linklookup(ctxt, ".got", 0);
			s->type = got->type | SSUB;
			s->outer = got;
			s->sub = got->sub;
			got->sub = s;
			s->value = got->size;
			adduint64(ctxt, got, 0);
			adduint32(ctxt, linklookup(ctxt, ".linkedit.got", 0), targ->dynid);
			r->type = 256;	// ignore during relocsym
			return;
		}
		break;
	}
	
	ctxt->cursym = s;
	diag("unsupported relocation for dynamic symbol %s (type=%d stype=%d)", targ->name, r->type, targ->type);
}

int
elfreloc1(Reloc *r, vlong sectoff)
{
	int32 elfsym;

	VPUT(sectoff);

	elfsym = r->xsym->elfsym;
	switch(r->type) {
	default:
		return -1;

	case R_ADDR:
		if(r->siz == 4)
			VPUT(R_X86_64_32 | (uint64)elfsym<<32);
		else if(r->siz == 8)
			VPUT(R_X86_64_64 | (uint64)elfsym<<32);
		else
			return -1;
		break;

	case R_TLS_LE:
		if(r->siz == 4)
			VPUT(R_X86_64_TPOFF32 | (uint64)elfsym<<32);
		else
			return -1;
		break;
		
	case R_CALL:
		if(r->siz == 4) {
			if(r->xsym->type == SDYNIMPORT)
				VPUT(R_X86_64_GOTPCREL | (uint64)elfsym<<32);
			else
				VPUT(R_X86_64_PC32 | (uint64)elfsym<<32);
		} else
			return -1;
		break;

	case R_PCREL:
		if(r->siz == 4) {
			VPUT(R_X86_64_PC32 | (uint64)elfsym<<32);
		} else
			return -1;
		break;

	case R_TLS:
		if(r->siz == 4) {
			if(flag_shared)
				VPUT(R_X86_64_GOTTPOFF | (uint64)elfsym<<32);
			else
				VPUT(R_X86_64_TPOFF32 | (uint64)elfsym<<32);
		} else
			return -1;
		break;		
	}

	VPUT(r->xadd);
	return 0;
}

int
machoreloc1(Reloc *r, vlong sectoff)
{
	uint32 v;
	LSym *rs;
	
	rs = r->xsym;

	if(rs->type == SHOSTOBJ || r->type == R_PCREL) {
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
	case R_ADDR:
		v |= MACHO_X86_64_RELOC_UNSIGNED<<28;
		break;
	case R_CALL:
		v |= 1<<24; // pc-relative bit
		v |= MACHO_X86_64_RELOC_BRANCH<<28;
		break;
	case R_PCREL:
		// NOTE: Only works with 'external' relocation. Forced above.
		v |= 1<<24; // pc-relative bit
		v |= MACHO_X86_64_RELOC_SIGNED<<28;
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
archreloc(Reloc *r, LSym *s, vlong *val)
{
	USED(r);
	USED(s);
	USED(val);
	return -1;
}

void
elfsetupplt(void)
{
	LSym *plt, *got;

	plt = linklookup(ctxt, ".plt", 0);
	got = linklookup(ctxt, ".got.plt", 0);
	if(plt->size == 0) {
		// pushq got+8(IP)
		adduint8(ctxt, plt, 0xff);
		adduint8(ctxt, plt, 0x35);
		addpcrelplus(ctxt, plt, got, 8);
		
		// jmpq got+16(IP)
		adduint8(ctxt, plt, 0xff);
		adduint8(ctxt, plt, 0x25);
		addpcrelplus(ctxt, plt, got, 16);
		
		// nopl 0(AX)
		adduint32(ctxt, plt, 0x00401f0f);
		
		// assume got->size == 0 too
		addaddrplus(ctxt, got, linklookup(ctxt, ".dynamic", 0), 0);
		adduint64(ctxt, got, 0);
		adduint64(ctxt, got, 0);
	}
}

static void
addpltsym(LSym *s)
{
	if(s->plt >= 0)
		return;
	
	adddynsym(ctxt, s);
	
	if(iself) {
		LSym *plt, *got, *rela;

		plt = linklookup(ctxt, ".plt", 0);
		got = linklookup(ctxt, ".got.plt", 0);
		rela = linklookup(ctxt, ".rela.plt", 0);
		if(plt->size == 0)
			elfsetupplt();
		
		// jmpq *got+size(IP)
		adduint8(ctxt, plt, 0xff);
		adduint8(ctxt, plt, 0x25);
		addpcrelplus(ctxt, plt, got, got->size);
	
		// add to got: pointer to current pos in plt
		addaddrplus(ctxt, got, plt, plt->size);
		
		// pushq $x
		adduint8(ctxt, plt, 0x68);
		adduint32(ctxt, plt, (got->size-24-8)/8);
		
		// jmpq .plt
		adduint8(ctxt, plt, 0xe9);
		adduint32(ctxt, plt, -(plt->size+4));
		
		// rela
		addaddrplus(ctxt, rela, got, got->size-8);
		adduint64(ctxt, rela, ELF64_R_INFO(s->dynid, R_X86_64_JMP_SLOT));
		adduint64(ctxt, rela, 0);
		
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

		LSym *plt;
		
		addgotsym(s);
		plt = linklookup(ctxt, ".plt", 0);

		adduint32(ctxt, linklookup(ctxt, ".linkedit.plt", 0), s->dynid);

		// jmpq *got+size(IP)
		s->plt = plt->size;

		adduint8(ctxt, plt, 0xff);
		adduint8(ctxt, plt, 0x25);
		addpcrelplus(ctxt, plt, linklookup(ctxt, ".got", 0), s->got);
	} else {
		diag("addpltsym: unsupported binary format");
	}
}

static void
addgotsym(LSym *s)
{
	LSym *got, *rela;

	if(s->got >= 0)
		return;

	adddynsym(ctxt, s);
	got = linklookup(ctxt, ".got", 0);
	s->got = got->size;
	adduint64(ctxt, got, 0);

	if(iself) {
		rela = linklookup(ctxt, ".rela", 0);
		addaddrplus(ctxt, rela, got, s->got);
		adduint64(ctxt, rela, ELF64_R_INFO(s->dynid, R_X86_64_GLOB_DAT));
		adduint64(ctxt, rela, 0);
	} else if(HEADTYPE == Hdarwin) {
		adduint32(ctxt, linklookup(ctxt, ".linkedit.got", 0), s->dynid);
	} else {
		diag("addgotsym: unsupported binary format");
	}
}

void
adddynsym(Link *ctxt, LSym *s)
{
	LSym *d;
	int t;
	char *name;

	if(s->dynid >= 0)
		return;

	if(iself) {
		s->dynid = nelfsym++;

		d = linklookup(ctxt, ".dynsym", 0);

		name = s->extname;
		adduint32(ctxt, d, addstring(linklookup(ctxt, ".dynstr", 0), name));
		/* type */
		t = STB_GLOBAL << 4;
		if(s->cgoexport && (s->type&SMASK) == STEXT)
			t |= STT_FUNC;
		else
			t |= STT_OBJECT;
		adduint8(ctxt, d, t);
	
		/* reserved */
		adduint8(ctxt, d, 0);
	
		/* section where symbol is defined */
		if(s->type == SDYNIMPORT)
			adduint16(ctxt, d, SHN_UNDEF);
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
			adduint16(ctxt, d, t);
		}
	
		/* value */
		if(s->type == SDYNIMPORT)
			adduint64(ctxt, d, 0);
		else
			addaddr(ctxt, d, s);
	
		/* size of object */
		adduint64(ctxt, d, s->size);
	
		if(!(s->cgoexport & CgoExportDynamic) && s->dynimplib && needlib(s->dynimplib)) {
			elfwritedynent(linklookup(ctxt, ".dynamic", 0), DT_NEEDED,
				addstring(linklookup(ctxt, ".dynstr", 0), s->dynimplib));
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
	LSym *s;
	
	if(!needlib(lib))
		return;
	
	if(iself) {
		s = linklookup(ctxt, ".dynstr", 0);
		if(s->size == 0)
			addstring(s, "");
		elfwritedynent(linklookup(ctxt, ".dynamic", 0), DT_NEEDED, addstring(s, lib));
	} else if(HEADTYPE == Hdarwin) {
		machoadddynlib(lib);
	} else {
		diag("adddynlib: unsupported binary format");
	}
}

void
asmb(void)
{
	int32 magic;
	int i;
	vlong vl, symo, dwarfoff, machlink;
	Section *sect;
	LSym *sym;

	if(debug['v'])
		Bprint(&bso, "%5.2f asmb\n", cputime());
	Bflush(&bso);

	if(debug['v'])
		Bprint(&bso, "%5.2f codeblk\n", cputime());
	Bflush(&bso);

	if(iself)
		asmbelfsetup();

	sect = segtext.sect;
	cseek(sect->vaddr - segtext.vaddr + segtext.fileoff);
	codeblk(sect->vaddr, sect->len);
	for(sect = sect->next; sect != nil; sect = sect->next) {
		cseek(sect->vaddr - segtext.vaddr + segtext.fileoff);
		datblk(sect->vaddr, sect->len);
	}

	if(segrodata.filelen > 0) {
		if(debug['v'])
			Bprint(&bso, "%5.2f rodatblk\n", cputime());
		Bflush(&bso);

		cseek(segrodata.fileoff);
		datblk(segrodata.vaddr, segrodata.filelen);
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
	case Hplan9:
	case Helf:
		break;
	case Hdarwin:
		debug['8'] = 1;	/* 64-bit addresses */
		break;
	case Hlinux:
	case Hfreebsd:
	case Hnetbsd:
	case Hopenbsd:
	case Hdragonfly:
	case Hsolaris:
		debug['8'] = 1;	/* 64-bit addresses */
		break;
	case Hnacl:
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
		case Hplan9:
		case Helf:
			debug['s'] = 1;
			symo = segdata.fileoff+segdata.filelen;
			break;
		case Hdarwin:
			symo = segdata.fileoff+rnd(segdata.filelen, INITRND)+machlink;
			break;
		case Hlinux:
		case Hfreebsd:
		case Hnetbsd:
		case Hopenbsd:
		case Hdragonfly:
		case Hsolaris:
		case Hnacl:
			symo = segdata.fileoff+segdata.filelen;
			symo = rnd(symo, INITRND);
			break;
		case Hwindows:
			symo = segdata.fileoff+segdata.filelen;
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
				
				if(linkmode == LinkExternal)
					elfemitreloc();
			}
			break;
		case Hplan9:
			asmplan9sym();
			cflush();

			sym = linklookup(ctxt, "pclntab", 0);
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
	case Hplan9:	/* plan9 */
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
	case Hdarwin:
		asmbmacho();
		break;
	case Hlinux:
	case Hfreebsd:
	case Hnetbsd:
	case Hopenbsd:
	case Hdragonfly:
	case Hsolaris:
	case Hnacl:
		asmbelf(symo);
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
