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
#include	"../ld/dwarf.h"

static Prog *PP;

char linuxdynld[] = "/lib/ld-linux.so.3"; // 2 for OABI, 3 for EABI
char freebsddynld[] = "/usr/libexec/ld-elf.so.1";
char openbsddynld[] = "XXX";
char netbsddynld[] = "/libexec/ld.elf_so";

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
static void	addgotsyminternal(Sym*);

// Preserve highest 8 bits of a, and do addition to lower 24-bit
// of a and b; used to adjust ARM branch intruction's target
static int32
braddoff(int32 a, int32 b)
{
	return (((uint32)a) & 0xff000000U) | (0x00ffffffU & (uint32)(a + b));
}

Sym *
lookuprel(void)
{
	return lookup(".rel", 0);
}

void
adddynrela(Sym *rel, Sym *s, Reloc *r)
{
	addaddrplus(rel, s, r->off);
	adduint32(rel, R_ARM_RELATIVE);
}

void
adddynrel(Sym *s, Reloc *r)
{
	Sym *targ, *rel;

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
	case 256 + R_ARM_PLT32:
		r->type = D_CALL;
		if(targ->type == SDYNIMPORT) {
			addpltsym(targ);
			r->sym = lookup(".plt", 0);
			r->add = braddoff(r->add, targ->plt / 4);
		}
		return;

	case 256 + R_ARM_THM_PC22: // R_ARM_THM_CALL
		diag("R_ARM_THM_CALL, are you using -marm?");
		errorexit();
		return;

	case 256 + R_ARM_GOT32: // R_ARM_GOT_BREL
		if(targ->type != SDYNIMPORT) {
			addgotsyminternal(targ);
		} else {
			addgotsym(targ);
		}
		r->type = D_CONST;	// write r->add during relocsym
		r->sym = S;
		r->add += targ->got;
		return;

	case 256 + R_ARM_GOT_PREL: // GOT(S) + A - P
		if(targ->type != SDYNIMPORT) {
			addgotsyminternal(targ);
		} else {
			addgotsym(targ);
		}
		r->type = D_PCREL;
		r->sym = lookup(".got", 0);
		r->add += targ->got + 4;
		return;

	case 256 + R_ARM_GOTOFF: // R_ARM_GOTOFF32
		r->type = D_GOTOFF;
		return;

	case 256 + R_ARM_GOTPC: // R_ARM_BASE_PREL
		r->type = D_PCREL;
		r->sym = lookup(".got", 0);
		r->add += 4;
		return;

	case 256 + R_ARM_CALL:
		r->type = D_CALL;
		if(targ->type == SDYNIMPORT) {
			addpltsym(targ);
			r->sym = lookup(".plt", 0);
			r->add = braddoff(r->add, targ->plt / 4);
		}
		return;

	case 256 + R_ARM_REL32: // R_ARM_REL32
		r->type = D_PCREL;
		r->add += 4;
		return;

	case 256 + R_ARM_ABS32: 
		if(targ->type == SDYNIMPORT)
			diag("unexpected R_ARM_ABS32 relocation for dynamic symbol %s", targ->name);
		r->type = D_ADDR;
		return;

	case 256 + R_ARM_V4BX:
		// we can just ignore this, because we are targeting ARM V5+ anyway
		if(r->sym) {
			// R_ARM_V4BX is ABS relocation, so this symbol is a dummy symbol, ignore it
			r->sym->type = 0;
		}
		r->sym = S;
		return;

	case 256 + R_ARM_PC24:
	case 256 + R_ARM_JUMP24:
		r->type = D_CALL;
		if(targ->type == SDYNIMPORT) {
			addpltsym(targ);
			r->sym = lookup(".plt", 0);
			r->add = braddoff(r->add, targ->plt / 4);
		}
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
			adduint32(rel, ELF32_R_INFO(targ->dynid, R_ARM_GLOB_DAT)); // we need a S + A dynmic reloc
			r->type = D_CONST;	// write r->add during relocsym
			r->sym = S;
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
			LPUT(R_ARM_ABS32 | elfsym<<8);
		else
			return -1;
		break;

	case D_PCREL:
		if(r->siz == 4)
			LPUT(R_ARM_REL32 | elfsym<<8);
		else
			return -1;
		break;
	}

	return 0;
}

void
elfsetupplt(void)
{
	Sym *plt, *got;
	
	plt = lookup(".plt", 0);
	got = lookup(".got.plt", 0);
	if(plt->size == 0) {
		// str lr, [sp, #-4]!
		adduint32(plt, 0xe52de004);
		// ldr lr, [pc, #4]
		adduint32(plt, 0xe59fe004);
		// add lr, pc, lr
		adduint32(plt, 0xe08fe00e);
		// ldr pc, [lr, #8]!
		adduint32(plt, 0xe5bef008);
		// .word &GLOBAL_OFFSET_TABLE[0] - .
		addpcrelplus(plt, got, 4);

		// the first .plt entry requires 3 .plt.got entries
		adduint32(got, 0);
		adduint32(got, 0);
		adduint32(got, 0);
	}
}

int
machoreloc1(Reloc *r, vlong sectoff)
{
	USED(r);
	USED(sectoff);

	return -1;
}


int
archreloc(Reloc *r, Sym *s, vlong *val)
{
	switch(r->type) {
	case D_CONST:
		*val = r->add;
		return 0;
	case D_GOTOFF:
		*val = symaddr(r->sym) + r->add - symaddr(lookup(".got", 0));
		return 0;
	// The following three arch specific relocations are only for generation of 
	// Linux/ARM ELF's PLT entry (3 assembler instruction)
	case D_PLT0: // add ip, pc, #0xXX00000
		if (symaddr(lookup(".got.plt", 0)) < symaddr(lookup(".plt", 0)))
			diag(".got.plt should be placed after .plt section.");
		*val = 0xe28fc600U +
			(0xff & ((uint32)(symaddr(r->sym) - (symaddr(lookup(".plt", 0)) + r->off) + r->add) >> 20));
		return 0;
	case D_PLT1: // add ip, ip, #0xYY000
		*val = 0xe28cca00U +
			(0xff & ((uint32)(symaddr(r->sym) - (symaddr(lookup(".plt", 0)) + r->off) + r->add + 4) >> 12));
		return 0;
	case D_PLT2: // ldr pc, [ip, #0xZZZ]!
		*val = 0xe5bcf000U +
			(0xfff & (uint32)(symaddr(r->sym) - (symaddr(lookup(".plt", 0)) + r->off) + r->add + 8));
		return 0;
	case D_CALL: // bl XXXXXX or b YYYYYY
		*val = braddoff((0xff000000U & (uint32)r->add), 
		                (0xffffff & (uint32)
		                   ((symaddr(r->sym) + ((uint32)r->add) * 4 - (s->value + r->off)) / 4)));
		return 0;
	}
	return -1;
}

static Reloc *
addpltreloc(Sym *plt, Sym *got, Sym *sym, int typ)
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
	symgrow(plt, plt->size);

	return r;
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
		
		// .got entry
		s->got = got->size;
		// In theory, all GOT should point to the first PLT entry,
		// Linux/ARM's dynamic linker will do that for us, but FreeBSD/ARM's
		// dynamic linker won't, so we'd better do it ourselves.
		addaddrplus(got, plt, 0);

		// .plt entry, this depends on the .got entry
		s->plt = plt->size;
		addpltreloc(plt, got, s, D_PLT0); // add lr, pc, #0xXX00000
		addpltreloc(plt, got, s, D_PLT1); // add lr, lr, #0xYY000
		addpltreloc(plt, got, s, D_PLT2); // ldr pc, [lr, #0xZZZ]!

		// rel
		addaddrplus(rel, got, s->got);
		adduint32(rel, ELF32_R_INFO(s->dynid, R_ARM_JUMP_SLOT));
	} else {
		diag("addpltsym: unsupported binary format");
	}
}

static void
addgotsyminternal(Sym *s)
{
	Sym *got;
	
	if(s->got >= 0)
		return;

	got = lookup(".got", 0);
	s->got = got->size;

	addaddrplus(got, s, 0);

	if(iself) {
		;
	} else {
		diag("addgotsyminternal: unsupported binary format");
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
		adduint32(rel, ELF32_R_INFO(s->dynid, R_ARM_GLOB_DAT));
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
		if((s->cgoexport & CgoExportDynamic) && (s->type&SMASK) == STEXT)
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
	} else {
		diag("adddynlib: unsupported binary format");
	}
}

vlong
datoff(vlong addr)
{
	if(addr >= segdata.vaddr)
		return addr - segdata.vaddr + segdata.fileoff;
	if(addr >= segtext.vaddr)
		return addr - segtext.vaddr + segtext.fileoff;
	diag("datoff %#x", addr);
	return 0;
}

void
asmb(void)
{
	int32 t;
	uint32 symo;
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
		case Hnoheader:
		case Hrisc:
		case Hixp1200:
		case Hipaq:
			debug['s'] = 1;
			break;
		case Hplan9x32:
			symo = HEADR+segtext.len+segdata.filelen;
			break;
		ElfSym:
			symo = rnd(HEADR+segtext.filelen, INITRND)+segdata.filelen;
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
		}
	}

	cursym = nil;
	if(debug['v'])
		Bprint(&bso, "%5.2f header\n", cputime());
	Bflush(&bso);
	cseek(0L);
	switch(HEADTYPE) {
	default:
	case Hnoheader:	/* no header */
		break;
	case Hrisc:	/* aif for risc os */
		lputl(0xe1a00000);		/* NOP - decompress code */
		lputl(0xe1a00000);		/* NOP - relocation code */
		lputl(0xeb000000 + 12);		/* BL - zero init code */
		lputl(0xeb000000 +
			(entryvalue()
			 - INITTEXT
			 + HEADR
			 - 12
			 - 8) / 4);		/* BL - entry code */

		lputl(0xef000011);		/* SWI - exit code */
		lputl(segtext.filelen+HEADR);		/* text size */
		lputl(segdata.filelen);			/* data size */
		lputl(0);			/* sym size */

		lputl(segdata.len - segdata.filelen);			/* bss size */
		lputl(0);			/* sym type */
		lputl(INITTEXT-HEADR);		/* text addr */
		lputl(0);			/* workspace - ignored */

		lputl(32);			/* addr mode / data addr flag */
		lputl(0);			/* data addr */
		for(t=0; t<2; t++)
			lputl(0);		/* reserved */

		for(t=0; t<15; t++)
			lputl(0xe1a00000);	/* NOP - zero init code */
		lputl(0xe1a0f00e);		/* B (R14) - zero init return */
		break;
	case Hplan9x32:	/* plan 9 */
		lput(0x647);			/* magic */
		lput(segtext.filelen);			/* sizes */
		lput(segdata.filelen);
		lput(segdata.len - segdata.filelen);
		lput(symsize);			/* nsyms */
		lput(entryvalue());		/* va of entry */
		lput(0L);
		lput(lcsize);
		break;
	case Hixp1200: /* boot for IXP1200 */
		break;
	case Hipaq: /* boot for ipaq */
		lputl(0xe3300000);		/* nop */
		lputl(0xe3300000);		/* nop */
		lputl(0xe3300000);		/* nop */
		lputl(0xe3300000);		/* nop */
		break;
	case Hlinux:
	case Hfreebsd:
	case Hnetbsd:
	case Hopenbsd:
		asmbelf(symo);
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

/*
void
cput(int32 c)
{
	*cbp++ = c;
	if(--cbc <= 0)
		cflush();
}
*/

void
wput(int32 l)
{

	cbp[0] = l>>8;
	cbp[1] = l;
	cbp += 2;
	cbc -= 2;
	if(cbc <= 0)
		cflush();
}


void
hput(int32 l)
{

	cbp[0] = l>>8;
	cbp[1] = l;
	cbp += 2;
	cbc -= 2;
	if(cbc <= 0)
		cflush();
}

void
lput(int32 l)
{

	cbp[0] = l>>24;
	cbp[1] = l>>16;
	cbp[2] = l>>8;
	cbp[3] = l;
	cbp += 4;
	cbc -= 4;
	if(cbc <= 0)
		cflush();
}

void
nopstat(char *f, Count *c)
{
	if(c->outof)
	Bprint(&bso, "%s delay %d/%d (%.2f)\n", f,
		c->outof - c->count, c->outof,
		(double)(c->outof - c->count)/c->outof);
}

void
asmout(Prog *p, Optab *o, int32 *out)
{
	int32 o1, o2, o3, o4, o5, o6, v;
	int r, rf, rt, rt2;
	Reloc *rel;

PP = p;
	o1 = 0;
	o2 = 0;
	o3 = 0;
	o4 = 0;
	o5 = 0;
	o6 = 0;
	armsize += o->size;
if(debug['P']) print("%ux: %P	type %d\n", (uint32)(p->pc), p, o->type);
	switch(o->type) {
	default:
		diag("unknown asm %d", o->type);
		prasm(p);
		break;

	case 0:		/* pseudo ops */
if(debug['G']) print("%ux: %s: arm %d\n", (uint32)(p->pc), p->from.sym->name, p->from.sym->fnptr);
		break;

	case 1:		/* op R,[R],R */
		o1 = oprrr(p->as, p->scond);
		rf = p->from.reg;
		rt = p->to.reg;
		r = p->reg;
		if(p->to.type == D_NONE)
			rt = 0;
		if(p->as == AMOVW || p->as == AMVN)
			r = 0;
		else
		if(r == NREG)
			r = rt;
		o1 |= rf | (r<<16) | (rt<<12);
		break;

	case 2:		/* movbu $I,[R],R */
		aclass(&p->from);
		o1 = oprrr(p->as, p->scond);
		o1 |= immrot(instoffset);
		rt = p->to.reg;
		r = p->reg;
		if(p->to.type == D_NONE)
			rt = 0;
		if(p->as == AMOVW || p->as == AMVN)
			r = 0;
		else if(r == NREG)
			r = rt;
		o1 |= (r<<16) | (rt<<12);
		break;

	case 3:		/* add R<<[IR],[R],R */
	mov:
		aclass(&p->from);
		o1 = oprrr(p->as, p->scond);
		o1 |= p->from.offset;
		rt = p->to.reg;
		r = p->reg;
		if(p->to.type == D_NONE)
			rt = 0;
		if(p->as == AMOVW || p->as == AMVN)
			r = 0;
		else if(r == NREG)
			r = rt;
		o1 |= (r<<16) | (rt<<12);
		break;

	case 4:		/* add $I,[R],R */
		aclass(&p->from);
		o1 = oprrr(AADD, p->scond);
		o1 |= immrot(instoffset);
		r = p->from.reg;
		if(r == NREG)
			r = o->param;
		o1 |= r << 16;
		o1 |= p->to.reg << 12;
		break;

	case 5:		/* bra s */
		v = -8;
		// TODO: Use addrel.
		if(p->cond != P)
			v = (p->cond->pc - pc) - 8;
		o1 = opbra(p->as, p->scond);
		o1 |= (v >> 2) & 0xffffff;
		break;

	case 6:		/* b ,O(R) -> add $O,R,PC */
		aclass(&p->to);
		o1 = oprrr(AADD, p->scond);
		o1 |= immrot(instoffset);
		o1 |= p->to.reg << 16;
		o1 |= REGPC << 12;
		break;

	case 7:		/* bl ,O(R) -> mov PC,link; add $O,R,PC */
		aclass(&p->to);
		o1 = oprrr(AADD, p->scond);
		o1 |= immrot(0);
		o1 |= REGPC << 16;
		o1 |= REGLINK << 12;

		o2 = oprrr(AADD, p->scond);
		o2 |= immrot(instoffset);
		o2 |= p->to.reg << 16;
		o2 |= REGPC << 12;
		break;

	case 8:		/* sll $c,[R],R -> mov (R<<$c),R */
		aclass(&p->from);
		o1 = oprrr(p->as, p->scond);
		r = p->reg;
		if(r == NREG)
			r = p->to.reg;
		o1 |= r;
		o1 |= (instoffset&31) << 7;
		o1 |= p->to.reg << 12;
		break;

	case 9:		/* sll R,[R],R -> mov (R<<R),R */
		o1 = oprrr(p->as, p->scond);
		r = p->reg;
		if(r == NREG)
			r = p->to.reg;
		o1 |= r;
		o1 |= (p->from.reg << 8) | (1<<4);
		o1 |= p->to.reg << 12;
		break;

	case 10:	/* swi [$con] */
		o1 = oprrr(p->as, p->scond);
		if(p->to.type != D_NONE) {
			aclass(&p->to);
			o1 |= instoffset & 0xffffff;
		}
		break;

	case 11:	/* word */
		aclass(&p->to);
		o1 = instoffset;
		if(p->to.sym != S) {
			rel = addrel(cursym);
			rel->off = pc - cursym->value;
			rel->siz = 4;
			rel->sym = p->to.sym;
			rel->add = p->to.offset;
			if(flag_shared) {
				rel->type = D_PCREL;
				rel->add += pc - p->pcrel->pc - 8;
			} else
				rel->type = D_ADDR;
			o1 = 0;
		}
		break;

	case 12:	/* movw $lcon, reg */
		o1 = omvl(p, &p->from, p->to.reg);
		if(o->flag & LPCREL) {
			o2 = oprrr(AADD, p->scond) | p->to.reg | REGPC << 16 | p->to.reg << 12;
		}
		break;

	case 13:	/* op $lcon, [R], R */
		o1 = omvl(p, &p->from, REGTMP);
		if(!o1)
			break;
		o2 = oprrr(p->as, p->scond);
		o2 |= REGTMP;
		r = p->reg;
		if(p->as == AMOVW || p->as == AMVN)
			r = 0;
		else if(r == NREG)
			r = p->to.reg;
		o2 |= r << 16;
		if(p->to.type != D_NONE)
			o2 |= p->to.reg << 12;
		break;

	case 14:	/* movb/movbu/movh/movhu R,R */
		o1 = oprrr(ASLL, p->scond);

		if(p->as == AMOVBU || p->as == AMOVHU)
			o2 = oprrr(ASRL, p->scond);
		else
			o2 = oprrr(ASRA, p->scond);

		r = p->to.reg;
		o1 |= (p->from.reg)|(r<<12);
		o2 |= (r)|(r<<12);
		if(p->as == AMOVB || p->as == AMOVBU) {
			o1 |= (24<<7);
			o2 |= (24<<7);
		} else {
			o1 |= (16<<7);
			o2 |= (16<<7);
		}
		break;

	case 15:	/* mul r,[r,]r */
		o1 = oprrr(p->as, p->scond);
		rf = p->from.reg;
		rt = p->to.reg;
		r = p->reg;
		if(r == NREG)
			r = rt;
		if(rt == r) {
			r = rf;
			rf = rt;
		}
		if(0)
		if(rt == r || rf == REGPC || r == REGPC || rt == REGPC) {
			diag("bad registers in MUL");
			prasm(p);
		}
		o1 |= (rf<<8) | r | (rt<<16);
		break;


	case 16:	/* div r,[r,]r */
		o1 = 0xf << 28;
		o2 = 0;
		break;

	case 17:
		o1 = oprrr(p->as, p->scond);
		rf = p->from.reg;
		rt = p->to.reg;
		rt2 = p->to.offset;
		r = p->reg;
		o1 |= (rf<<8) | r | (rt<<16) | (rt2<<12);
		break;

	case 20:	/* mov/movb/movbu R,O(R) */
		aclass(&p->to);
		r = p->to.reg;
		if(r == NREG)
			r = o->param;
		o1 = osr(p->as, p->from.reg, instoffset, r, p->scond);
		break;

	case 21:	/* mov/movbu O(R),R -> lr */
		aclass(&p->from);
		r = p->from.reg;
		if(r == NREG)
			r = o->param;
		o1 = olr(instoffset, r, p->to.reg, p->scond);
		if(p->as != AMOVW)
			o1 |= 1<<22;
		break;

	case 30:	/* mov/movb/movbu R,L(R) */
		o1 = omvl(p, &p->to, REGTMP);
		if(!o1)
			break;
		r = p->to.reg;
		if(r == NREG)
			r = o->param;
		o2 = osrr(p->from.reg, REGTMP,r, p->scond);
		if(p->as != AMOVW)
			o2 |= 1<<22;
		break;

	case 31:	/* mov/movbu L(R),R -> lr[b] */
		o1 = omvl(p, &p->from, REGTMP);
		if(!o1)
			break;
		r = p->from.reg;
		if(r == NREG)
			r = o->param;
		o2 = olrr(REGTMP,r, p->to.reg, p->scond);
		if(p->as == AMOVBU || p->as == AMOVB)
			o2 |= 1<<22;
		break;

	case 34:	/* mov $lacon,R */
		o1 = omvl(p, &p->from, REGTMP);
		if(!o1)
			break;

		o2 = oprrr(AADD, p->scond);
		o2 |= REGTMP;
		r = p->from.reg;
		if(r == NREG)
			r = o->param;
		o2 |= r << 16;
		if(p->to.type != D_NONE)
			o2 |= p->to.reg << 12;
		break;

	case 35:	/* mov PSR,R */
		o1 = (2<<23) | (0xf<<16) | (0<<0);
		o1 |= (p->scond & C_SCOND) << 28;
		o1 |= (p->from.reg & 1) << 22;
		o1 |= p->to.reg << 12;
		break;

	case 36:	/* mov R,PSR */
		o1 = (2<<23) | (0x29f<<12) | (0<<4);
		if(p->scond & C_FBIT)
			o1 ^= 0x010 << 12;
		o1 |= (p->scond & C_SCOND) << 28;
		o1 |= (p->to.reg & 1) << 22;
		o1 |= p->from.reg << 0;
		break;

	case 37:	/* mov $con,PSR */
		aclass(&p->from);
		o1 = (2<<23) | (0x29f<<12) | (0<<4);
		if(p->scond & C_FBIT)
			o1 ^= 0x010 << 12;
		o1 |= (p->scond & C_SCOND) << 28;
		o1 |= immrot(instoffset);
		o1 |= (p->to.reg & 1) << 22;
		o1 |= p->from.reg << 0;
		break;

	case 38:	/* movm $con,oreg -> stm */
		o1 = (0x4 << 25);
		o1 |= p->from.offset & 0xffff;
		o1 |= p->to.reg << 16;
		aclass(&p->to);
		goto movm;

	case 39:	/* movm oreg,$con -> ldm */
		o1 = (0x4 << 25) | (1 << 20);
		o1 |= p->to.offset & 0xffff;
		o1 |= p->from.reg << 16;
		aclass(&p->from);
	movm:
		if(instoffset != 0)
			diag("offset must be zero in MOVM");
		o1 |= (p->scond & C_SCOND) << 28;
		if(p->scond & C_PBIT)
			o1 |= 1 << 24;
		if(p->scond & C_UBIT)
			o1 |= 1 << 23;
		if(p->scond & C_SBIT)
			o1 |= 1 << 22;
		if(p->scond & C_WBIT)
			o1 |= 1 << 21;
		break;

	case 40:	/* swp oreg,reg,reg */
		aclass(&p->from);
		if(instoffset != 0)
			diag("offset must be zero in SWP");
		o1 = (0x2<<23) | (0x9<<4);
		if(p->as != ASWPW)
			o1 |= 1 << 22;
		o1 |= p->from.reg << 16;
		o1 |= p->reg << 0;
		o1 |= p->to.reg << 12;
		o1 |= (p->scond & C_SCOND) << 28;
		break;

	case 41:	/* rfe -> movm.s.w.u 0(r13),[r15] */
		o1 = 0xe8fd8000;
		break;

	case 50:	/* floating point store */
		v = regoff(&p->to);
		r = p->to.reg;
		if(r == NREG)
			r = o->param;
		o1 = ofsr(p->as, p->from.reg, v, r, p->scond, p);
		break;

	case 51:	/* floating point load */
		v = regoff(&p->from);
		r = p->from.reg;
		if(r == NREG)
			r = o->param;
		o1 = ofsr(p->as, p->to.reg, v, r, p->scond, p) | (1<<20);
		break;

	case 52:	/* floating point store, int32 offset UGLY */
		o1 = omvl(p, &p->to, REGTMP);
		if(!o1)
			break;
		r = p->to.reg;
		if(r == NREG)
			r = o->param;
		o2 = oprrr(AADD, p->scond) | (REGTMP << 12) | (REGTMP << 16) | r;
		o3 = ofsr(p->as, p->from.reg, 0, REGTMP, p->scond, p);
		break;

	case 53:	/* floating point load, int32 offset UGLY */
		o1 = omvl(p, &p->from, REGTMP);
		if(!o1)
			break;
		r = p->from.reg;
		if(r == NREG)
			r = o->param;
		o2 = oprrr(AADD, p->scond) | (REGTMP << 12) | (REGTMP << 16) | r;
		o3 = ofsr(p->as, p->to.reg, 0, REGTMP, p->scond, p) | (1<<20);
		break;

	case 54:	/* floating point arith */
		o1 = oprrr(p->as, p->scond);
		rf = p->from.reg;
		rt = p->to.reg;
		r = p->reg;
		if(r == NREG) {
			r = rt;
			if(p->as == AMOVF || p->as == AMOVD || p->as == ASQRTF || p->as == ASQRTD || p->as == AABSF || p->as == AABSD)
				r = 0;
		}
		o1 |= rf | (r<<16) | (rt<<12);
		break;

	case 56:	/* move to FP[CS]R */
		o1 = ((p->scond & C_SCOND) << 28) | (0xe << 24) | (1<<8) | (1<<4);
		o1 |= ((p->to.reg+1)<<21) | (p->from.reg << 12);
		break;

	case 57:	/* move from FP[CS]R */
		o1 = ((p->scond & C_SCOND) << 28) | (0xe << 24) | (1<<8) | (1<<4);
		o1 |= ((p->from.reg+1)<<21) | (p->to.reg<<12) | (1<<20);
		break;
	case 58:	/* movbu R,R */
		o1 = oprrr(AAND, p->scond);
		o1 |= immrot(0xff);
		rt = p->to.reg;
		r = p->from.reg;
		if(p->to.type == D_NONE)
			rt = 0;
		if(r == NREG)
			r = rt;
		o1 |= (r<<16) | (rt<<12);
		break;

	case 59:	/* movw/bu R<<I(R),R -> ldr indexed */
		if(p->from.reg == NREG) {
			if(p->as != AMOVW)
				diag("byte MOV from shifter operand");
			goto mov;
		}
		if(p->from.offset&(1<<4))
			diag("bad shift in LDR");
		o1 = olrr(p->from.offset, p->from.reg, p->to.reg, p->scond);
		if(p->as == AMOVBU)
			o1 |= 1<<22;
		break;

	case 60:	/* movb R(R),R -> ldrsb indexed */
		if(p->from.reg == NREG) {
			diag("byte MOV from shifter operand");
			goto mov;
		}
		if(p->from.offset&(~0xf))
			diag("bad shift in LDRSB");
		o1 = olhrr(p->from.offset, p->from.reg, p->to.reg, p->scond);
		o1 ^= (1<<5)|(1<<6);
		break;

	case 61:	/* movw/b/bu R,R<<[IR](R) -> str indexed */
		if(p->to.reg == NREG)
			diag("MOV to shifter operand");
		o1 = osrr(p->from.reg, p->to.offset, p->to.reg, p->scond);
		if(p->as == AMOVB || p->as == AMOVBU)
			o1 |= 1<<22;
		break;

	case 62:	/* case R -> movw	R<<2(PC),PC */
		if(o->flag & LPCREL) {
			o1 = oprrr(AADD, p->scond) | immrot(1) | p->from.reg << 16 | REGTMP << 12;
			o2 = olrr(REGTMP, REGPC, REGTMP, p->scond);
			o2 |= 2<<7;
			o3 = oprrr(AADD, p->scond) | REGTMP | REGPC << 16 | REGPC << 12;
		} else {
			o1 = olrr(p->from.reg, REGPC, REGPC, p->scond);
			o1 |= 2<<7;
		}
		break;

	case 63:	/* bcase */
		if(p->cond != P) {
			o1 = p->cond->pc;
			if(flag_shared)
				o1 = o1 - p->pcrel->pc - 16;
		}
		break;

	/* reloc ops */
	case 64:	/* mov/movb/movbu R,addr */
		o1 = omvl(p, &p->to, REGTMP);
		if(!o1)
			break;
		o2 = osr(p->as, p->from.reg, 0, REGTMP, p->scond);
		if(o->flag & LPCREL) {
			o3 = o2;
			o2 = oprrr(AADD, p->scond) | REGTMP | REGPC << 16 | REGTMP << 12;
		}
		break;

	case 65:	/* mov/movbu addr,R */
		o1 = omvl(p, &p->from, REGTMP);
		if(!o1)
			break;
		o2 = olr(0, REGTMP, p->to.reg, p->scond);
		if(p->as == AMOVBU || p->as == AMOVB)
			o2 |= 1<<22;
		if(o->flag & LPCREL) {
			o3 = o2;
			o2 = oprrr(AADD, p->scond) | REGTMP | REGPC << 16 | REGTMP << 12;
		}
		break;

	case 68:	/* floating point store -> ADDR */
		o1 = omvl(p, &p->to, REGTMP);
		if(!o1)
			break;
		o2 = ofsr(p->as, p->from.reg, 0, REGTMP, p->scond, p);
		if(o->flag & LPCREL) {
			o3 = o2;
			o2 = oprrr(AADD, p->scond) | REGTMP | REGPC << 16 | REGTMP << 12;
		}
		break;

	case 69:	/* floating point load <- ADDR */
		o1 = omvl(p, &p->from, REGTMP);
		if(!o1)
			break;
		o2 = ofsr(p->as, p->to.reg, 0, REGTMP, p->scond, p) | (1<<20);
		if(o->flag & LPCREL) {
			o3 = o2;
			o2 = oprrr(AADD, p->scond) | REGTMP | REGPC << 16 | REGTMP << 12;
		}
		break;

	/* ArmV4 ops: */
	case 70:	/* movh/movhu R,O(R) -> strh */
		aclass(&p->to);
		r = p->to.reg;
		if(r == NREG)
			r = o->param;
		o1 = oshr(p->from.reg, instoffset, r, p->scond);
		break;
	case 71:	/* movb/movh/movhu O(R),R -> ldrsb/ldrsh/ldrh */
		aclass(&p->from);
		r = p->from.reg;
		if(r == NREG)
			r = o->param;
		o1 = olhr(instoffset, r, p->to.reg, p->scond);
		if(p->as == AMOVB)
			o1 ^= (1<<5)|(1<<6);
		else if(p->as == AMOVH)
			o1 ^= (1<<6);
		break;
	case 72:	/* movh/movhu R,L(R) -> strh */
		o1 = omvl(p, &p->to, REGTMP);
		if(!o1)
			break;
		r = p->to.reg;
		if(r == NREG)
			r = o->param;
		o2 = oshrr(p->from.reg, REGTMP,r, p->scond);
		break;
	case 73:	/* movb/movh/movhu L(R),R -> ldrsb/ldrsh/ldrh */
		o1 = omvl(p, &p->from, REGTMP);
		if(!o1)
			break;
		r = p->from.reg;
		if(r == NREG)
			r = o->param;
		o2 = olhrr(REGTMP, r, p->to.reg, p->scond);
		if(p->as == AMOVB)
			o2 ^= (1<<5)|(1<<6);
		else if(p->as == AMOVH)
			o2 ^= (1<<6);
		break;
	case 74:	/* bx $I */
		diag("ABX $I");
		break;
	case 75:	/* bx O(R) */
		aclass(&p->to);
		if(instoffset != 0)
			diag("non-zero offset in ABX");
/*
		o1 = 	oprrr(AADD, p->scond) | immrot(0) | (REGPC<<16) | (REGLINK<<12);	// mov PC, LR
		o2 = ((p->scond&C_SCOND)<<28) | (0x12fff<<8) | (1<<4) | p->to.reg;		// BX R
*/
		// p->to.reg may be REGLINK
		o1 = oprrr(AADD, p->scond);
		o1 |= immrot(instoffset);
		o1 |= p->to.reg << 16;
		o1 |= REGTMP << 12;
		o2 = oprrr(AADD, p->scond) | immrot(0) | (REGPC<<16) | (REGLINK<<12);	// mov PC, LR
		o3 = ((p->scond&C_SCOND)<<28) | (0x12fff<<8) | (1<<4) | REGTMP;		// BX Rtmp
		break;
	case 76:	/* bx O(R) when returning from fn*/
		diag("ABXRET");
		break;
	case 77:	/* ldrex oreg,reg */
		aclass(&p->from);
		if(instoffset != 0)
			diag("offset must be zero in LDREX");
		o1 = (0x19<<20) | (0xf9f);
		o1 |= p->from.reg << 16;
		o1 |= p->to.reg << 12;
		o1 |= (p->scond & C_SCOND) << 28;
		break;
	case 78:	/* strex reg,oreg,reg */
		aclass(&p->from);
		if(instoffset != 0)
			diag("offset must be zero in STREX");
		o1 = (0x18<<20) | (0xf90);
		o1 |= p->from.reg << 16;
		o1 |= p->reg << 0;
		o1 |= p->to.reg << 12;
		o1 |= (p->scond & C_SCOND) << 28;
		break;
	case 80:	/* fmov zfcon,freg */
		if(p->as == AMOVD) {
			o1 = 0xeeb00b00;	// VMOV imm 64
			o2 = oprrr(ASUBD, p->scond);
		} else {
			o1 = 0x0eb00a00;	// VMOV imm 32
			o2 = oprrr(ASUBF, p->scond);
		}
		v = 0x70;	// 1.0
		r = p->to.reg;

		// movf $1.0, r
		o1 |= (p->scond & C_SCOND) << 28;
		o1 |= r << 12;
		o1 |= (v&0xf) << 0;
		o1 |= (v&0xf0) << 12;

		// subf r,r,r
		o2 |= r | (r<<16) | (r<<12);
		break;
	case 81:	/* fmov sfcon,freg */
		o1 = 0x0eb00a00;		// VMOV imm 32
		if(p->as == AMOVD)
			o1 = 0xeeb00b00;	// VMOV imm 64
		o1 |= (p->scond & C_SCOND) << 28;
		o1 |= p->to.reg << 12;
		v = chipfloat(&p->from.ieee);
		o1 |= (v&0xf) << 0;
		o1 |= (v&0xf0) << 12;
		break;
	case 82:	/* fcmp freg,freg, */
		o1 = oprrr(p->as, p->scond);
		o1 |= (p->reg<<12) | (p->from.reg<<0);
		o2 = 0x0ef1fa10;	// VMRS R15
		o2 |= (p->scond & C_SCOND) << 28;
		break;
	case 83:	/* fcmp freg,, */
		o1 = oprrr(p->as, p->scond);
		o1 |= (p->from.reg<<12) | (1<<16);
		o2 = 0x0ef1fa10;	// VMRS R15
		o2 |= (p->scond & C_SCOND) << 28;
		break;
	case 84:	/* movfw freg,freg - truncate float-to-fix */
		o1 = oprrr(p->as, p->scond);
		o1 |= (p->from.reg<<0);
		o1 |= (p->to.reg<<12);
		break;
	case 85:	/* movwf freg,freg - fix-to-float */
		o1 = oprrr(p->as, p->scond);
		o1 |= (p->from.reg<<0);
		o1 |= (p->to.reg<<12);
		break;
	case 86:	/* movfw freg,reg - truncate float-to-fix */
		// macro for movfw freg,FTMP; movw FTMP,reg
		o1 = oprrr(p->as, p->scond);
		o1 |= (p->from.reg<<0);
		o1 |= (FREGTMP<<12);
		o2 = oprrr(AMOVFW+AEND, p->scond);
		o2 |= (FREGTMP<<16);
		o2 |= (p->to.reg<<12);
		break;
	case 87:	/* movwf reg,freg - fix-to-float */
		// macro for movw reg,FTMP; movwf FTMP,freg
		o1 = oprrr(AMOVWF+AEND, p->scond);
		o1 |= (p->from.reg<<12);
		o1 |= (FREGTMP<<16);
		o2 = oprrr(p->as, p->scond);
		o2 |= (FREGTMP<<0);
		o2 |= (p->to.reg<<12);
		break;
	case 88:	/* movw reg,freg  */
		o1 = oprrr(AMOVWF+AEND, p->scond);
		o1 |= (p->from.reg<<12);
		o1 |= (p->to.reg<<16);
		break;
	case 89:	/* movw freg,reg  */
		o1 = oprrr(AMOVFW+AEND, p->scond);
		o1 |= (p->from.reg<<16);
		o1 |= (p->to.reg<<12);
		break;
	case 90:	/* tst reg  */
		o1 = oprrr(ACMP+AEND, p->scond);
		o1 |= p->from.reg<<16;
		break;
	case 91:	/* ldrexd oreg,reg */
		aclass(&p->from);
		if(instoffset != 0)
			diag("offset must be zero in LDREX");
		o1 = (0x1b<<20) | (0xf9f);
		o1 |= p->from.reg << 16;
		o1 |= p->to.reg << 12;
		o1 |= (p->scond & C_SCOND) << 28;
		break;
	case 92:	/* strexd reg,oreg,reg */
		aclass(&p->from);
		if(instoffset != 0)
			diag("offset must be zero in STREX");
		o1 = (0x1a<<20) | (0xf90);
		o1 |= p->from.reg << 16;
		o1 |= p->reg << 0;
		o1 |= p->to.reg << 12;
		o1 |= (p->scond & C_SCOND) << 28;
		break;
	case 93:	/* movb/movh/movhu addr,R -> ldrsb/ldrsh/ldrh */
		o1 = omvl(p, &p->from, REGTMP);
		if(!o1)
			break;
		o2 = olhr(0, REGTMP, p->to.reg, p->scond);
		if(p->as == AMOVB)
			o2 ^= (1<<5)|(1<<6);
		else if(p->as == AMOVH)
			o2 ^= (1<<6);
		if(o->flag & LPCREL) {
			o3 = o2;
			o2 = oprrr(AADD, p->scond) | REGTMP | REGPC << 16 | REGTMP << 12;
		}
		break;
	case 94:	/* movh/movhu R,addr -> strh */
		o1 = omvl(p, &p->to, REGTMP);
		if(!o1)
			break;
		o2 = oshr(p->from.reg, 0, REGTMP, p->scond);
		if(o->flag & LPCREL) {
			o3 = o2;
			o2 = oprrr(AADD, p->scond) | REGTMP | REGPC << 16 | REGTMP << 12;
		}
		break;
	case 95:	/* PLD off(reg) */
		o1 = 0xf5d0f000;
		o1 |= p->from.reg << 16;
		if(p->from.offset < 0) {
			o1 &= ~(1 << 23);
			o1 |= (-p->from.offset) & 0xfff;
		} else
			o1 |= p->from.offset & 0xfff;
		break;
	case 96:	/* UNDEF */
		// This is supposed to be something that stops execution.
		// It's not supposed to be reached, ever, but if it is, we'd
		// like to be able to tell how we got there.  Assemble as
		//	BL $0
		// TODO: Use addrel.
		v = (0 - pc) - 8;
		o1 = opbra(ABL, C_SCOND_NONE);
		o1 |= (v >> 2) & 0xffffff;
		break;
	case 97:	/* CLZ Rm, Rd */
 		o1 = oprrr(p->as, p->scond);
 		o1 |= p->to.reg << 12;
 		o1 |= p->from.reg;
		break;
	case 98:	/* MULW{T,B} Rs, Rm, Rd */
		o1 = oprrr(p->as, p->scond);
		o1 |= p->to.reg << 16;
		o1 |= p->from.reg << 8;
		o1 |= p->reg;
		break;
	case 99:	/* MULAW{T,B} Rs, Rm, Rn, Rd */
		o1 = oprrr(p->as, p->scond);
		o1 |= p->to.reg << 12;
		o1 |= p->from.reg << 8;
		o1 |= p->reg;
		o1 |= p->to.offset << 16;
		break;
	}
	
	out[0] = o1;
	out[1] = o2;
	out[2] = o3;
	out[3] = o4;
	out[4] = o5;
	out[5] = o6;
	return;

#ifdef NOTDEF
	v = p->pc;
	switch(o->size) {
	default:
		if(debug['a'])
			Bprint(&bso, " %.8ux:\t\t%P\n", v, p);
		break;
	case 4:
		if(debug['a'])
			Bprint(&bso, " %.8ux: %.8ux\t%P\n", v, o1, p);
		lputl(o1);
		break;
	case 8:
		if(debug['a'])
			Bprint(&bso, " %.8ux: %.8ux %.8ux%P\n", v, o1, o2, p);
		lputl(o1);
		lputl(o2);
		break;
	case 12:
		if(debug['a'])
			Bprint(&bso, " %.8ux: %.8ux %.8ux %.8ux%P\n", v, o1, o2, o3, p);
		lputl(o1);
		lputl(o2);
		lputl(o3);
		break;
	case 16:
		if(debug['a'])
			Bprint(&bso, " %.8ux: %.8ux %.8ux %.8ux %.8ux%P\n",
				v, o1, o2, o3, o4, p);
		lputl(o1);
		lputl(o2);
		lputl(o3);
		lputl(o4);
		break;
	case 20:
		if(debug['a'])
			Bprint(&bso, " %.8ux: %.8ux %.8ux %.8ux %.8ux %.8ux%P\n",
				v, o1, o2, o3, o4, o5, p);
		lputl(o1);
		lputl(o2);
		lputl(o3);
		lputl(o4);
		lputl(o5);
		break;
	case 24:
		if(debug['a'])
			Bprint(&bso, " %.8ux: %.8ux %.8ux %.8ux %.8ux %.8ux %.8ux%P\n",
				v, o1, o2, o3, o4, o5, o6, p);
		lputl(o1);
		lputl(o2);
		lputl(o3);
		lputl(o4);
		lputl(o5);
		lputl(o6);
		break;
	}
#endif
}

int32
oprrr(int a, int sc)
{
	int32 o;

	o = (sc & C_SCOND) << 28;
	if(sc & C_SBIT)
		o |= 1 << 20;
	if(sc & (C_PBIT|C_WBIT))
		diag(".P/.W on dp instruction");
	switch(a) {
	case AMULU:
	case AMUL:	return o | (0x0<<21) | (0x9<<4);
	case AMULA:	return o | (0x1<<21) | (0x9<<4);
	case AMULLU:	return o | (0x4<<21) | (0x9<<4);
	case AMULL:	return o | (0x6<<21) | (0x9<<4);
	case AMULALU:	return o | (0x5<<21) | (0x9<<4);
	case AMULAL:	return o | (0x7<<21) | (0x9<<4);
	case AAND:	return o | (0x0<<21);
	case AEOR:	return o | (0x1<<21);
	case ASUB:	return o | (0x2<<21);
	case ARSB:	return o | (0x3<<21);
	case AADD:	return o | (0x4<<21);
	case AADC:	return o | (0x5<<21);
	case ASBC:	return o | (0x6<<21);
	case ARSC:	return o | (0x7<<21);
	case ATST:	return o | (0x8<<21) | (1<<20);
	case ATEQ:	return o | (0x9<<21) | (1<<20);
	case ACMP:	return o | (0xa<<21) | (1<<20);
	case ACMN:	return o | (0xb<<21) | (1<<20);
	case AORR:	return o | (0xc<<21);
	case AMOVW:	return o | (0xd<<21);
	case ABIC:	return o | (0xe<<21);
	case AMVN:	return o | (0xf<<21);
	case ASLL:	return o | (0xd<<21) | (0<<5);
	case ASRL:	return o | (0xd<<21) | (1<<5);
	case ASRA:	return o | (0xd<<21) | (2<<5);
	case ASWI:	return o | (0xf<<24);

	case AADDD:	return o | (0xe<<24) | (0x3<<20) | (0xb<<8) | (0<<4);
	case AADDF:	return o | (0xe<<24) | (0x3<<20) | (0xa<<8) | (0<<4);
	case ASUBD:	return o | (0xe<<24) | (0x3<<20) | (0xb<<8) | (4<<4);
	case ASUBF:	return o | (0xe<<24) | (0x3<<20) | (0xa<<8) | (4<<4);
	case AMULD:	return o | (0xe<<24) | (0x2<<20) | (0xb<<8) | (0<<4);
	case AMULF:	return o | (0xe<<24) | (0x2<<20) | (0xa<<8) | (0<<4);
	case ADIVD:	return o | (0xe<<24) | (0x8<<20) | (0xb<<8) | (0<<4);
	case ADIVF:	return o | (0xe<<24) | (0x8<<20) | (0xa<<8) | (0<<4);
	case ASQRTD:	return o | (0xe<<24) | (0xb<<20) | (1<<16) | (0xb<<8) | (0xc<<4);
	case ASQRTF:	return o | (0xe<<24) | (0xb<<20) | (1<<16) | (0xa<<8) | (0xc<<4);
	case AABSD:	return o | (0xe<<24) | (0xb<<20) | (0<<16) | (0xb<<8) | (0xc<<4);
	case AABSF:	return o | (0xe<<24) | (0xb<<20) | (0<<16) | (0xa<<8) | (0xc<<4);
	case ACMPD:	return o | (0xe<<24) | (0xb<<20) | (4<<16) | (0xb<<8) | (0xc<<4);
	case ACMPF:	return o | (0xe<<24) | (0xb<<20) | (4<<16) | (0xa<<8) | (0xc<<4);

	case AMOVF:	return o | (0xe<<24) | (0xb<<20) | (0<<16) | (0xa<<8) | (4<<4);
	case AMOVD:	return o | (0xe<<24) | (0xb<<20) | (0<<16) | (0xb<<8) | (4<<4);

	case AMOVDF:	return o | (0xe<<24) | (0xb<<20) | (7<<16) | (0xa<<8) | (0xc<<4) |
			(1<<8);	// dtof
	case AMOVFD:	return o | (0xe<<24) | (0xb<<20) | (7<<16) | (0xa<<8) | (0xc<<4) |
			(0<<8);	// dtof

	case AMOVWF:
			if((sc & C_UBIT) == 0)
				o |= 1<<7;	/* signed */
			return o | (0xe<<24) | (0xb<<20) | (8<<16) | (0xa<<8) | (4<<4) |
				(0<<18) | (0<<8);	// toint, double
	case AMOVWD:
			if((sc & C_UBIT) == 0)
				o |= 1<<7;	/* signed */
			return o | (0xe<<24) | (0xb<<20) | (8<<16) | (0xa<<8) | (4<<4) |
				(0<<18) | (1<<8);	// toint, double

	case AMOVFW:
			if((sc & C_UBIT) == 0)
				o |= 1<<16;	/* signed */
			return o | (0xe<<24) | (0xb<<20) | (8<<16) | (0xa<<8) | (4<<4) |
				(1<<18) | (0<<8) | (1<<7);	// toint, double, trunc
	case AMOVDW:
			if((sc & C_UBIT) == 0)
				o |= 1<<16;	/* signed */
			return o | (0xe<<24) | (0xb<<20) | (8<<16) | (0xa<<8) | (4<<4) |
				(1<<18) | (1<<8) | (1<<7);	// toint, double, trunc

	case AMOVWF+AEND:	// copy WtoF
		return o | (0xe<<24) | (0x0<<20) | (0xb<<8) | (1<<4);
	case AMOVFW+AEND:	// copy FtoW
		return o | (0xe<<24) | (0x1<<20) | (0xb<<8) | (1<<4);
	case ACMP+AEND:	// cmp imm
		return o | (0x3<<24) | (0x5<<20);

	case ACLZ:
		// CLZ doesn't support .S
		return (o & (0xf<<28)) | (0x16f<<16) | (0xf1<<4);

	case AMULWT:
		return (o & (0xf<<28)) | (0x12 << 20) | (0xe<<4);
	case AMULWB:
		return (o & (0xf<<28)) | (0x12 << 20) | (0xa<<4);
	case AMULAWT:
		return (o & (0xf<<28)) | (0x12 << 20) | (0xc<<4);
	case AMULAWB:
		return (o & (0xf<<28)) | (0x12 << 20) | (0x8<<4);
	}
	diag("bad rrr %d", a);
	prasm(curp);
	return 0;
}

int32
opbra(int a, int sc)
{

	if(sc & (C_SBIT|C_PBIT|C_WBIT))
		diag(".S/.P/.W on bra instruction");
	sc &= C_SCOND;
	if(a == ABL)
		return (sc<<28)|(0x5<<25)|(0x1<<24);
	if(sc != 0xe)
		diag(".COND on bcond instruction");
	switch(a) {
	case ABEQ:	return (0x0<<28)|(0x5<<25);
	case ABNE:	return (0x1<<28)|(0x5<<25);
	case ABCS:	return (0x2<<28)|(0x5<<25);
	case ABHS:	return (0x2<<28)|(0x5<<25);
	case ABCC:	return (0x3<<28)|(0x5<<25);
	case ABLO:	return (0x3<<28)|(0x5<<25);
	case ABMI:	return (0x4<<28)|(0x5<<25);
	case ABPL:	return (0x5<<28)|(0x5<<25);
	case ABVS:	return (0x6<<28)|(0x5<<25);
	case ABVC:	return (0x7<<28)|(0x5<<25);
	case ABHI:	return (0x8<<28)|(0x5<<25);
	case ABLS:	return (0x9<<28)|(0x5<<25);
	case ABGE:	return (0xa<<28)|(0x5<<25);
	case ABLT:	return (0xb<<28)|(0x5<<25);
	case ABGT:	return (0xc<<28)|(0x5<<25);
	case ABLE:	return (0xd<<28)|(0x5<<25);
	case AB:	return (0xe<<28)|(0x5<<25);
	}
	diag("bad bra %A", a);
	prasm(curp);
	return 0;
}

int32
olr(int32 v, int b, int r, int sc)
{
	int32 o;

	if(sc & C_SBIT)
		diag(".S on LDR/STR instruction");
	o = (sc & C_SCOND) << 28;
	if(!(sc & C_PBIT))
		o |= 1 << 24;
	if(!(sc & C_UBIT))
		o |= 1 << 23;
	if(sc & C_WBIT)
		o |= 1 << 21;
	o |= (1<<26) | (1<<20);
	if(v < 0) {
		if(sc & C_UBIT) diag(".U on neg offset");
		v = -v;
		o ^= 1 << 23;
	}
	if(v >= (1<<12) || v < 0)
		diag("literal span too large: %d (R%d)\n%P", v, b, PP);
	o |= v;
	o |= b << 16;
	o |= r << 12;
	return o;
}

int32
olhr(int32 v, int b, int r, int sc)
{
	int32 o;

	if(sc & C_SBIT)
		diag(".S on LDRH/STRH instruction");
	o = (sc & C_SCOND) << 28;
	if(!(sc & C_PBIT))
		o |= 1 << 24;
	if(sc & C_WBIT)
		o |= 1 << 21;
	o |= (1<<23) | (1<<20)|(0xb<<4);
	if(v < 0) {
		v = -v;
		o ^= 1 << 23;
	}
	if(v >= (1<<8) || v < 0)
		diag("literal span too large: %d (R%d)\n%P", v, b, PP);
	o |= (v&0xf)|((v>>4)<<8)|(1<<22);
	o |= b << 16;
	o |= r << 12;
	return o;
}

int32
osr(int a, int r, int32 v, int b, int sc)
{
	int32 o;

	o = olr(v, b, r, sc) ^ (1<<20);
	if(a != AMOVW)
		o |= 1<<22;
	return o;
}

int32
oshr(int r, int32 v, int b, int sc)
{
	int32 o;

	o = olhr(v, b, r, sc) ^ (1<<20);
	return o;
}


int32
osrr(int r, int i, int b, int sc)
{

	return olr(i, b, r, sc) ^ ((1<<25) | (1<<20));
}

int32
oshrr(int r, int i, int b, int sc)
{
	return olhr(i, b, r, sc) ^ ((1<<22) | (1<<20));
}

int32
olrr(int i, int b, int r, int sc)
{

	return olr(i, b, r, sc) ^ (1<<25);
}

int32
olhrr(int i, int b, int r, int sc)
{
	return olhr(i, b, r, sc) ^ (1<<22);
}

int32
ofsr(int a, int r, int32 v, int b, int sc, Prog *p)
{
	int32 o;

	if(sc & C_SBIT)
		diag(".S on FLDR/FSTR instruction");
	o = (sc & C_SCOND) << 28;
	if(!(sc & C_PBIT))
		o |= 1 << 24;
	if(sc & C_WBIT)
		o |= 1 << 21;
	o |= (6<<25) | (1<<24) | (1<<23) | (10<<8);
	if(v < 0) {
		v = -v;
		o ^= 1 << 23;
	}
	if(v & 3)
		diag("odd offset for floating point op: %d\n%P", v, p);
	else
	if(v >= (1<<10) || v < 0)
		diag("literal span too large: %d\n%P", v, p);
	o |= (v>>2) & 0xFF;
	o |= b << 16;
	o |= r << 12;

	switch(a) {
	default:
		diag("bad fst %A", a);
	case AMOVD:
		o |= 1 << 8;
	case AMOVF:
		break;
	}
	return o;
}

int32
omvl(Prog *p, Adr *a, int dr)
{
	int32 v, o1;
	if(!p->cond) {
		aclass(a);
		v = immrot(~instoffset);
		if(v == 0) {
			diag("missing literal");
			prasm(p);
			return 0;
		}
		o1 = oprrr(AMVN, p->scond&C_SCOND);
		o1 |= v;
		o1 |= dr << 12;
	} else {
		v = p->cond->pc - p->pc - 8;
		o1 = olr(v, REGPC, dr, p->scond&C_SCOND);
	}
	return o1;
}

int
chipzero(Ieee *e)
{
	// We use GOARM=7 to gate the use of VFPv3 vmov (imm) instructions.
	if(goarm < 7 || e->l != 0 || e->h != 0)
		return -1;
	return 0;
}

int
chipfloat(Ieee *e)
{
	int n;
	ulong h;

	// We use GOARM=7 to gate the use of VFPv3 vmov (imm) instructions.
	if(goarm < 7)
		goto no;

	if(e->l != 0 || (e->h&0xffff) != 0)
		goto no;
	h = e->h & 0x7fc00000;
	if(h != 0x40000000 && h != 0x3fc00000)
		goto no;
	n = 0;

	// sign bit (a)
	if(e->h & 0x80000000)
		n |= 1<<7;

	// exp sign bit (b)
	if(h == 0x3fc00000)
		n |= 1<<6;

	// rest of exp and mantissa (cd-efgh)
	n |= (e->h >> 16) & 0x3f;

//print("match %.8lux %.8lux %d\n", e->l, e->h, n);
	return n;

no:
	return -1;
}
