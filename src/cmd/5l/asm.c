// Inferno utils/5l/asm.c
// http://code.google.com/p/inferno-os/source/browse/utils/5l/asm.c
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
#include	"../ld/macho.h"
#include	"../ld/dwarf.h"

static int
needlib(char *name)
{
	char *p;
	LSym *s;

	if(*name == '\0')
		return 0;

	/* reuse hash code in symbol table */
	p = smprint(".dynlib.%s", name);
	s = linklookup(ctxt, p, 0);
	free(p);
	if(s->type == 0) {
		s->type = 100;	// avoid SDATA, etc.
		return 1;
	}
	return 0;
}

static void	addpltsym(Link*, LSym*);
static void	addgotsym(Link*, LSym*);
static void	addgotsyminternal(Link*, LSym*);

void
gentext(void)
{
}

// Preserve highest 8 bits of a, and do addition to lower 24-bit
// of a and b; used to adjust ARM branch intruction's target
static int32
braddoff(int32 a, int32 b)
{
	return (((uint32)a) & 0xff000000U) | (0x00ffffffU & (uint32)(a + b));
}

void
adddynrela(LSym *rel, LSym *s, Reloc *r)
{
	addaddrplus(ctxt, rel, s, r->off);
	adduint32(ctxt, rel, R_ARM_RELATIVE);
}

void
adddynrel(LSym *s, Reloc *r)
{
	LSym *targ, *rel;

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
	case 256 + R_ARM_PLT32:
		r->type = R_CALLARM;
		if(targ->type == SDYNIMPORT) {
			addpltsym(ctxt, targ);
			r->sym = linklookup(ctxt, ".plt", 0);
			r->add = braddoff(r->add, targ->plt / 4);
		}
		return;

	case 256 + R_ARM_THM_PC22: // R_ARM_THM_CALL
		diag("R_ARM_THM_CALL, are you using -marm?");
		errorexit();
		return;

	case 256 + R_ARM_GOT32: // R_ARM_GOT_BREL
		if(targ->type != SDYNIMPORT) {
			addgotsyminternal(ctxt, targ);
		} else {
			addgotsym(ctxt, targ);
		}
		r->type = R_CONST;	// write r->add during relocsym
		r->sym = nil;
		r->add += targ->got;
		return;

	case 256 + R_ARM_GOT_PREL: // GOT(nil) + A - nil
		if(targ->type != SDYNIMPORT) {
			addgotsyminternal(ctxt, targ);
		} else {
			addgotsym(ctxt, targ);
		}
		r->type = R_PCREL;
		r->sym = linklookup(ctxt, ".got", 0);
		r->add += targ->got + 4;
		return;

	case 256 + R_ARM_GOTOFF: // R_ARM_GOTOFF32
		r->type = R_GOTOFF;
		return;

	case 256 + R_ARM_GOTPC: // R_ARM_BASE_PREL
		r->type = R_PCREL;
		r->sym = linklookup(ctxt, ".got", 0);
		r->add += 4;
		return;

	case 256 + R_ARM_CALL:
		r->type = R_CALLARM;
		if(targ->type == SDYNIMPORT) {
			addpltsym(ctxt, targ);
			r->sym = linklookup(ctxt, ".plt", 0);
			r->add = braddoff(r->add, targ->plt / 4);
		}
		return;

	case 256 + R_ARM_REL32: // R_ARM_REL32
		r->type = R_PCREL;
		r->add += 4;
		return;

	case 256 + R_ARM_ABS32: 
		if(targ->type == SDYNIMPORT)
			diag("unexpected R_ARM_ABS32 relocation for dynamic symbol %s", targ->name);
		r->type = R_ADDR;
		return;

	case 256 + R_ARM_V4BX:
		// we can just ignore this, because we are targeting ARM V5+ anyway
		if(r->sym) {
			// R_ARM_V4BX is ABS relocation, so this symbol is a dummy symbol, ignore it
			r->sym->type = 0;
		}
		r->sym = nil;
		return;

	case 256 + R_ARM_PC24:
	case 256 + R_ARM_JUMP24:
		r->type = R_CALLARM;
		if(targ->type == SDYNIMPORT) {
			addpltsym(ctxt, targ);
			r->sym = linklookup(ctxt, ".plt", 0);
			r->add = braddoff(r->add, targ->plt / 4);
		}
		return;
	}
	
	// Handle references to ELF symbols from our own object files.
	if(targ->type != SDYNIMPORT)
		return;

	switch(r->type) {
	case R_CALLARM:
		addpltsym(ctxt, targ);
		r->sym = linklookup(ctxt, ".plt", 0);
		r->add = targ->plt;
		return;
	
	case R_ADDR:
		if(s->type != SDATA)
			break;
		if(iself) {
			adddynsym(ctxt, targ);
			rel = linklookup(ctxt, ".rel", 0);
			addaddrplus(ctxt, rel, s, r->off);
			adduint32(ctxt, rel, ELF32_R_INFO(targ->dynid, R_ARM_GLOB_DAT)); // we need a nil + A dynmic reloc
			r->type = R_CONST;	// write r->add during relocsym
			r->sym = nil;
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
	
	thearch.lput(sectoff);

	elfsym = r->xsym->elfsym;
	switch(r->type) {
	default:
		return -1;

	case R_ADDR:
		if(r->siz == 4)
			thearch.lput(R_ARM_ABS32 | elfsym<<8);
		else
			return -1;
		break;

	case R_PCREL:
		if(r->siz == 4)
			thearch.lput(R_ARM_REL32 | elfsym<<8);
		else
			return -1;
		break;

	case R_CALLARM:
		if(r->siz == 4) {
			if((r->add & 0xff000000) == 0xeb000000) // BL
				thearch.lput(R_ARM_CALL | elfsym<<8);
			else
				thearch.lput(R_ARM_JUMP24 | elfsym<<8);
		} else
			return -1;
		break;

	case R_TLS:
		if(r->siz == 4) {
			if(flag_shared)
				thearch.lput(R_ARM_TLS_IE32 | elfsym<<8);
			else
				thearch.lput(R_ARM_TLS_LE32 | elfsym<<8);
		} else
			return -1;
		break;
	}

	return 0;
}

void
elfsetupplt(void)
{
	LSym *plt, *got;
	
	plt = linklookup(ctxt, ".plt", 0);
	got = linklookup(ctxt, ".got.plt", 0);
	if(plt->size == 0) {
		// str lr, [sp, #-4]!
		adduint32(ctxt, plt, 0xe52de004);
		// ldr lr, [pc, #4]
		adduint32(ctxt, plt, 0xe59fe004);
		// add lr, pc, lr
		adduint32(ctxt, plt, 0xe08fe00e);
		// ldr pc, [lr, #8]!
		adduint32(ctxt, plt, 0xe5bef008);
		// .word &GLOBAL_OFFSET_TABLE[0] - .
		addpcrelplus(ctxt, plt, got, 4);

		// the first .plt entry requires 3 .plt.got entries
		adduint32(ctxt, got, 0);
		adduint32(ctxt, got, 0);
		adduint32(ctxt, got, 0);
	}
}

int
machoreloc1(Reloc *r, vlong sectoff)
{
	uint32 v;
	LSym *rs;

	rs = r->xsym;

	if(rs->type == SHOSTOBJ || r->type == R_CALLARM) {
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
		v |= MACHO_GENERIC_RELOC_VANILLA<<28;
		break;
	case R_CALLARM:
		v |= 1<<24; // pc-relative bit
		v |= MACHO_ARM_RELOC_BR24<<28;
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

	thearch.lput(sectoff);
	thearch.lput(v);
	return 0;
}


int
archreloc(Reloc *r, LSym *s, vlong *val)
{
	LSym *rs;

	if(linkmode == LinkExternal) {
		switch(r->type) {
		case R_CALLARM:
			r->done = 0;

			// set up addend for eventual relocation via outer symbol.
			rs = r->sym;
			r->xadd = r->add;
			if(r->xadd & 0x800000)
				r->xadd |= ~0xffffff;
			r->xadd *= 4;
			while(rs->outer != nil) {
				r->xadd += symaddr(rs) - symaddr(rs->outer);
				rs = rs->outer;
			}

			if(rs->type != SHOSTOBJ && rs->sect == nil)
				diag("missing section for %s", rs->name);
			r->xsym = rs;

			// ld64 for arm seems to want the symbol table to contain offset
			// into the section rather than pseudo virtual address that contains
			// the section load address.
			// we need to compensate that by removing the instruction's address
			// from addend.
			if(HEADTYPE == Hdarwin)
				r->xadd -= symaddr(s) + r->off;

			*val = braddoff((0xff000000U & (uint32)r->add), 
							(0xffffff & (uint32)(r->xadd / 4)));
			return 0;
		}
		return -1;
	}
	switch(r->type) {
	case R_CONST:
		*val = r->add;
		return 0;
	case R_GOTOFF:
		*val = symaddr(r->sym) + r->add - symaddr(linklookup(ctxt, ".got", 0));
		return 0;
	// The following three arch specific relocations are only for generation of 
	// Linux/ARM ELF's PLT entry (3 assembler instruction)
	case R_PLT0: // add ip, pc, #0xXX00000
		if (symaddr(linklookup(ctxt, ".got.plt", 0)) < symaddr(linklookup(ctxt, ".plt", 0)))
			diag(".got.plt should be placed after .plt section.");
		*val = 0xe28fc600U +
			(0xff & ((uint32)(symaddr(r->sym) - (symaddr(linklookup(ctxt, ".plt", 0)) + r->off) + r->add) >> 20));
		return 0;
	case R_PLT1: // add ip, ip, #0xYY000
		*val = 0xe28cca00U +
			(0xff & ((uint32)(symaddr(r->sym) - (symaddr(linklookup(ctxt, ".plt", 0)) + r->off) + r->add + 4) >> 12));
		return 0;
	case R_PLT2: // ldr pc, [ip, #0xZZZ]!
		*val = 0xe5bcf000U +
			(0xfff & (uint32)(symaddr(r->sym) - (symaddr(linklookup(ctxt, ".plt", 0)) + r->off) + r->add + 8));
		return 0;
	case R_CALLARM: // bl XXXXXX or b YYYYYY
		*val = braddoff((0xff000000U & (uint32)r->add), 
		                (0xffffff & (uint32)
		                   ((symaddr(r->sym) + ((uint32)r->add) * 4 - (s->value + r->off)) / 4)));
		return 0;
	}
	return -1;
}

vlong
archrelocvariant(Reloc *r, LSym *s, vlong t)
{
	USED(r);
	USED(s);
	sysfatal("unexpected relocation variant");
	return t;
}

static Reloc *
addpltreloc(Link *ctxt, LSym *plt, LSym *got, LSym *sym, int typ)
{
	Reloc *r;

	r = addrel(plt);
	r->sym = got;
	r->off = plt->size;
	r->siz = 4;
	r->type = typ;
	r->add = sym->got - 8;

	plt->reachable = 1;
	plt->size += 4;
	symgrow(ctxt, plt, plt->size);

	return r;
}

static void
addpltsym(Link *ctxt, LSym *s)
{
	LSym *plt, *got, *rel;
	
	if(s->plt >= 0)
		return;

	adddynsym(ctxt, s);
	
	if(iself) {
		plt = linklookup(ctxt, ".plt", 0);
		got = linklookup(ctxt, ".got.plt", 0);
		rel = linklookup(ctxt, ".rel.plt", 0);
		if(plt->size == 0)
			elfsetupplt();
		
		// .got entry
		s->got = got->size;
		// In theory, all GOT should point to the first PLT entry,
		// Linux/ARM's dynamic linker will do that for us, but FreeBSD/ARM's
		// dynamic linker won't, so we'd better do it ourselves.
		addaddrplus(ctxt, got, plt, 0);

		// .plt entry, this depends on the .got entry
		s->plt = plt->size;
		addpltreloc(ctxt, plt, got, s, R_PLT0); // add lr, pc, #0xXX00000
		addpltreloc(ctxt, plt, got, s, R_PLT1); // add lr, lr, #0xYY000
		addpltreloc(ctxt, plt, got, s, R_PLT2); // ldr pc, [lr, #0xZZZ]!

		// rel
		addaddrplus(ctxt, rel, got, s->got);
		adduint32(ctxt, rel, ELF32_R_INFO(s->dynid, R_ARM_JUMP_SLOT));
	} else {
		diag("addpltsym: unsupported binary format");
	}
}

static void
addgotsyminternal(Link *ctxt, LSym *s)
{
	LSym *got;
	
	if(s->got >= 0)
		return;

	got = linklookup(ctxt, ".got", 0);
	s->got = got->size;

	addaddrplus(ctxt, got, s, 0);

	if(iself) {
		;
	} else {
		diag("addgotsyminternal: unsupported binary format");
	}
}

static void
addgotsym(Link *ctxt, LSym *s)
{
	LSym *got, *rel;
	
	if(s->got >= 0)
		return;
	
	adddynsym(ctxt, s);
	got = linklookup(ctxt, ".got", 0);
	s->got = got->size;
	adduint32(ctxt, got, 0);
	
	if(iself) {
		rel = linklookup(ctxt, ".rel", 0);
		addaddrplus(ctxt, rel, got, s->got);
		adduint32(ctxt, rel, ELF32_R_INFO(s->dynid, R_ARM_GLOB_DAT));
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

		/* name */
		name = s->extname;
		adduint32(ctxt, d, addstring(linklookup(ctxt, ".dynstr", 0), name));

		/* value */
		if(s->type == SDYNIMPORT)
			adduint32(ctxt, d, 0);
		else
			addaddr(ctxt, d, s);

		/* size */
		adduint32(ctxt, d, 0);

		/* type */
		t = STB_GLOBAL << 4;
		if((s->cgoexport & CgoExportDynamic) && (s->type&SMASK) == STEXT)
			t |= STT_FUNC;
		else
			t |= STT_OBJECT;
		adduint8(ctxt, d, t);
		adduint8(ctxt, d, 0);

		/* shndx */
		if(s->type == SDYNIMPORT)
			adduint16(ctxt, d, SHN_UNDEF);
		else
			adduint16(ctxt, d, 1);
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
	uint32 symo, dwarfoff, machlink;
	Section *sect;
	LSym *sym;
	int i;

	if(debug['v'])
		Bprint(&bso, "%5.2f asmb\n", cputime());
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

		if(!debug['w']) { // TODO(minux): enable DWARF Support
			dwarfoff = rnd(HEADR+segtext.len, INITRND) + rnd(segdata.filelen, INITRND);
			cseek(dwarfoff);

			segdwarf.fileoff = cpos();
			dwarfemitdebugsections();
			segdwarf.filelen = cpos() - segdwarf.fileoff;
		}
		machlink = domacholink();
	}

	/* output symbol table */
	symsize = 0;
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
				goto ElfSym;
		case Hplan9:
			symo = segdata.fileoff+segdata.filelen;
			break;
		case Hdarwin:
			symo = rnd(HEADR+segtext.filelen, INITRND)+rnd(segdata.filelen, INITRND)+machlink;
			break;
		ElfSym:
			symo = segdata.fileoff+segdata.filelen;
			symo = rnd(symo, INITRND);
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
		case Hdarwin:
			if(linkmode == LinkExternal)
				machoemitreloc();
			break;
		}
	}

	ctxt->cursym = nil;
	if(debug['v'])
		Bprint(&bso, "%5.2f header\n", cputime());
	Bflush(&bso);
	cseek(0L);
	switch(HEADTYPE) {
	default:
	case Hplan9:	/* plan 9 */
		thearch.lput(0x647);			/* magic */
		thearch.lput(segtext.filelen);			/* sizes */
		thearch.lput(segdata.filelen);
		thearch.lput(segdata.len - segdata.filelen);
		thearch.lput(symsize);			/* nsyms */
		thearch.lput(entryvalue());		/* va of entry */
		thearch.lput(0L);
		thearch.lput(lcsize);
		break;
	case Hlinux:
	case Hfreebsd:
	case Hnetbsd:
	case Hopenbsd:
	case Hnacl:
		asmbelf(symo);
		break;
	case Hdarwin:
		asmbmacho();
		break;
	}
	cflush();
	if(debug['c']){
		print("textsize=%ulld\n", segtext.filelen);
		print("datsize=%ulld\n", segdata.filelen);
		print("bsssize=%ulld\n", segdata.len - segdata.filelen);
		print("symsize=%d\n", symsize);
		print("lcsize=%d\n", lcsize);
		print("total=%lld\n", segtext.filelen+segdata.len+symsize+lcsize);
	}
}
