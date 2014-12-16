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


// TODO(austin): ABI v1 uses /usr/lib/ld.so.1
char linuxdynld[] = "/lib64/ld64.so.1";
char freebsddynld[] = "XXX";
char openbsddynld[] = "XXX";
char netbsddynld[] = "XXX";
char dragonflydynld[] = "XXX";
char solarisdynld[] = "XXX";

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

int	nelfsym = 1;

static void	gencallstub(int abicase, LSym *stub, LSym *targ);
static void	addpltsym(Link*, LSym*);
static LSym*	ensureglinkresolver(void);

void
gentext(void)
{
	LSym *s, *stub, **pprevtextp;
	Reloc *r;
	char *n;
	uint32 o1;
	uchar *cast;
	int i;

	// The ppc64 ABI PLT has similar concepts to other
	// architectures, but is laid out quite differently.  When we
	// see an R_PPC64_REL24 relocation to a dynamic symbol
	// (indicating that the call needs to go through the PLT), we
	// generate up to three stubs and reserve a PLT slot.
	//
	// 1) The call site will be bl x; nop (where the relocation
	//    applies to the bl).  We rewrite this to bl x_stub; ld
	//    r2,24(r1).  The ld is necessary because x_stub will save
	//    r2 (the TOC pointer) at 24(r1) (the "TOC save slot").
	//
	// 2) We reserve space for a pointer in the .plt section (once
	//    per referenced dynamic function).  .plt is a data
	//    section filled solely by the dynamic linker (more like
	//    .plt.got on other architectures).  Initially, the
	//    dynamic linker will fill each slot with a pointer to the
	//    corresponding x@plt entry point.
	//
	// 3) We generate the "call stub" x_stub (once per dynamic
	//    function/object file pair).  This saves the TOC in the
	//    TOC save slot, reads the function pointer from x's .plt
	//    slot and calls it like any other global entry point
	//    (including setting r12 to the function address).
	//
	// 4) We generate the "symbol resolver stub" x@plt (once per
	//    dynamic function).  This is solely a branch to the glink
	//    resolver stub.
	//
	// 5) We generate the glink resolver stub (only once).  This
	//    computes which symbol resolver stub we came through and
	//    invokes the dynamic resolver via a pointer provided by
	//    the dynamic linker.  This will patch up the .plt slot to
	//    point directly at the function so future calls go
	//    straight from the call stub to the real function, and
	//    then call the function.

	// NOTE: It's possible we could make ppc64 closer to other
	// architectures: ppc64's .plt is like .plt.got on other
	// platforms and ppc64's .glink is like .plt on other
	// platforms.

	// Find all R_PPC64_REL24 relocations that reference dynamic
	// imports.  Reserve PLT entries for these symbols and
	// generate call stubs.  The call stubs need to live in .text,
	// which is why we need to do this pass this early.
	//
	// This assumes "case 1" from the ABI, where the caller needs
	// us to save and restore the TOC pointer.
	pprevtextp = &ctxt->textp;
	for(s=*pprevtextp; s!=S; pprevtextp=&s->next, s=*pprevtextp) {
		for(r=s->r; r<s->r+s->nr; r++) {
			if(!(r->type == 256 + R_PPC64_REL24 &&
			     r->sym->type == SDYNIMPORT))
				continue;

			// Reserve PLT entry and generate symbol
			// resolver
			addpltsym(ctxt, r->sym);

			// Generate call stub
			n = smprint("%s.%s", s->name, r->sym->name);
			stub = linklookup(ctxt, n, 0);
			free(n);
			stub->reachable |= s->reachable;
			if(stub->size == 0) {
				// Need outer to resolve .TOC.
				stub->outer = s;

				// Link in to textp before s (we could
				// do it after, but would have to skip
				// the subsymbols)
				*pprevtextp = stub;
				stub->next = s;
				pprevtextp = &stub->next;

				gencallstub(1, stub, r->sym);
			}

			// Update the relocation to use the call stub
			r->sym = stub;

			// Restore TOC after bl.  The compiler put a
			// nop here for us to overwrite.
			o1 = 0xe8410018; // ld r2,24(r1)
			cast = (uchar*)&o1;
			for(i=0; i<4; i++)
				s->p[r->off+4+i] = cast[inuxi4[i]];
		}
	}
}

// Construct a call stub in stub that calls symbol targ via its PLT
// entry.
static void
gencallstub(int abicase, LSym *stub, LSym *targ)
{
	LSym *plt;
	Reloc *r;

	if(abicase != 1)
		// If we see R_PPC64_TOCSAVE or R_PPC64_REL24_NOTOC
		// relocations, we'll need to implement cases 2 and 3.
		sysfatal("gencallstub only implements case 1 calls");

	plt = linklookup(ctxt, ".plt", 0);

	stub->type = STEXT;

	// Save TOC pointer in TOC save slot
	adduint32(ctxt, stub, 0xf8410018); // std r2,24(r1)

	// Load the function pointer from the PLT.
	r = addrel(stub);
	r->off = stub->size;
	r->sym = plt;
	r->add = targ->plt;
	r->siz = 2;
	if(ctxt->arch->endian == BigEndian)
		r->off += r->siz;
	r->type = R_POWER_TOC;
	r->variant = RV_POWER_HA;
	adduint32(ctxt, stub, 0x3d820000); // addis r12,r2,targ@plt@toc@ha
	r = addrel(stub);
	r->off = stub->size;
	r->sym = plt;
	r->add = targ->plt;
	r->siz = 2;
	if(ctxt->arch->endian == BigEndian)
		r->off += r->siz;
	r->type = R_POWER_TOC;
	r->variant = RV_POWER_LO;
	adduint32(ctxt, stub, 0xe98c0000); // ld r12,targ@plt@toc@l(r12)

	// Jump to the loaded pointer
	adduint32(ctxt, stub, 0x7d8903a6); // mtctr r12
	adduint32(ctxt, stub, 0x4e800420); // bctr
}

void
adddynrela(LSym *rel, LSym *s, Reloc *r)
{
	USED(rel); USED(s); USED(r);
	sysfatal("adddynrela not implemented");
}

void
adddynrel(LSym *s, Reloc *r)
{
	LSym *targ, *rela;

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
	case 256 + R_PPC64_REL24:
		r->type = R_CALLPOWER;
		// This is a local call, so the caller isn't setting
		// up r12 and r2 is the same for the caller and
		// callee.  Hence, we need to go to the local entry
		// point.  (If we don't do this, the callee will try
		// to use r12 to compute r2.)
		r->add += r->sym->localentry * 4;
		if(targ->type == SDYNIMPORT)
			// Should have been handled in elfsetupplt
			diag("unexpected R_PPC64_REL24 for dyn import");
		return;

	case 256 + R_PPC64_ADDR64:
		r->type = R_ADDR;
		if(targ->type == SDYNIMPORT) {
			// These happen in .toc sections
			adddynsym(ctxt, targ);

			rela = linklookup(ctxt, ".rela", 0);
			addaddrplus(ctxt, rela, s, r->off);
			adduint64(ctxt, rela, ELF64_R_INFO(targ->dynid, R_PPC64_ADDR64));
			adduint64(ctxt, rela, r->add);
			r->type = 256;	// ignore during relocsym
		}
		return;

	case 256 + R_PPC64_TOC16:
		r->type = R_POWER_TOC;
		r->variant = RV_POWER_LO | RV_CHECK_OVERFLOW;
		return;

	case 256 + R_PPC64_TOC16_LO:
		r->type = R_POWER_TOC;
		r->variant = RV_POWER_LO;
		return;

	case 256 + R_PPC64_TOC16_HA:
		r->type = R_POWER_TOC;
		r->variant = RV_POWER_HA | RV_CHECK_OVERFLOW;
		return;

	case 256 + R_PPC64_TOC16_HI:
		r->type = R_POWER_TOC;
		r->variant = RV_POWER_HI | RV_CHECK_OVERFLOW;
		return;

	case 256 + R_PPC64_TOC16_DS:
		r->type = R_POWER_TOC;
		r->variant = RV_POWER_DS | RV_CHECK_OVERFLOW;
		return;

	case 256 + R_PPC64_TOC16_LO_DS:
		r->type = R_POWER_TOC;
		r->variant = RV_POWER_DS;
		return;

	case 256 + R_PPC64_REL16_LO:
		r->type = R_PCREL;
		r->variant = RV_POWER_LO;
		r->add += 2;	// Compensate for relocation size of 2
		return;

	case 256 + R_PPC64_REL16_HI:
		r->type = R_PCREL;
		r->variant = RV_POWER_HI | RV_CHECK_OVERFLOW;
		r->add += 2;
		return;

	case 256 + R_PPC64_REL16_HA:
		r->type = R_PCREL;
		r->variant = RV_POWER_HA | RV_CHECK_OVERFLOW;
		r->add += 2;
		return;
	}

	// Handle references to ELF symbols from our own object files.
	if(targ->type != SDYNIMPORT)
		return;

	// TODO(austin): Translate our relocations to ELF

	diag("unsupported relocation for dynamic symbol %s (type=%d stype=%d)", targ->name, r->type, targ->type);
}

int
elfreloc1(Reloc *r, vlong sectoff)
{
	USED(r); USED(sectoff);
	// TODO(minux)
	return -1;
}

void
elfsetupplt(void)
{
	LSym *plt;

	plt = linklookup(ctxt, ".plt", 0);
	if(plt->size == 0) {
		// The dynamic linker stores the address of the
		// dynamic resolver and the DSO identifier in the two
		// doublewords at the beginning of the .plt section
		// before the PLT array.  Reserve space for these.
		plt->size = 16;
	}
}

int
machoreloc1(Reloc *r, vlong sectoff)
{
	USED(r);
	USED(sectoff);

	return -1;
}

// Return the value of .TOC. for symbol s
static vlong
symtoc(LSym *s)
{
	LSym *toc;

	if(s->outer != nil)
		toc = linkrlookup(ctxt, ".TOC.", s->outer->version);
	else
		toc = linkrlookup(ctxt, ".TOC.", s->version);

	if(toc == nil) {
		diag("TOC-relative relocation in object without .TOC.");
		return 0;
	}
	return toc->value;
}

int
archreloc(Reloc *r, LSym *s, vlong *val)
{
	uint32 o1, o2;
	vlong t;

	if(linkmode == LinkExternal) {
		// TODO(minux): translate R_ADDRPOWER and R_CALLPOWER into standard ELF relocations.
		// R_ADDRPOWER corresponds to R_PPC_ADDR16_HA and R_PPC_ADDR16_LO.
		// R_CALLPOWER corresponds to R_PPC_REL24.
		return -1;
	}
	switch(r->type) {
	case R_CONST:
		*val = r->add;
		return 0;
	case R_GOTOFF:
		*val = symaddr(r->sym) + r->add - symaddr(linklookup(ctxt, ".got", 0));
		return 0;
	case R_ADDRPOWER:
		// r->add is two ppc64 instructions holding an immediate 32-bit constant.
		// We want to add r->sym's address to that constant.
		// The encoding of the immediate x<<16 + y,
		// where x is the low 16 bits of the first instruction and y is the low 16
		// bits of the second. Both x and y are signed (int16, not uint16).
		o1 = r->add >> 32;
		o2 = r->add;
		t = symaddr(r->sym);
		if(t < 0) {
			ctxt->diag("relocation for %s is too big (>=2G): %lld", s->name, symaddr(r->sym));
		}
		t += ((o1 & 0xffff) << 16) + ((int32)o2 << 16 >> 16);
		if(t & 0x8000)
			t += 0x10000;
		o1 = (o1 & 0xffff0000) | ((t >> 16) & 0xffff);
		o2 = (o2 & 0xffff0000) | (t & 0xffff);
		// when laid out, the instruction order must always be o1, o2.
		if(ctxt->arch->endian == BigEndian)
			*val = ((vlong)o1 << 32) | o2;
		else
			*val = ((vlong)o2 << 32) | o1;
		return 0;
	case R_CALLPOWER:
		// Bits 6 through 29 = (S + A - P) >> 2
		if(ctxt->arch->endian == BigEndian)
			o1 = be32(s->p + r->off);
		else
			o1 = le32(s->p + r->off);

		t = symaddr(r->sym) + r->add - (s->value + r->off);
		if(t & 3)
			ctxt->diag("relocation for %s+%d is not aligned: %lld", r->sym->name, r->off, t);
		if((int32)(t << 6) >> 6 != t)
			// TODO(austin) This can happen if text > 32M.
			// Add a call trampoline to .text in that case.
			ctxt->diag("relocation for %s+%d is too big: %lld", r->sym->name, r->off, t);

		*val = (o1 & 0xfc000003U) | (t & ~0xfc000003U);
		return 0;
	case R_POWER_TOC:	// S + A - .TOC.
		*val = symaddr(r->sym) + r->add - symtoc(s);
		return 0;
	}
	return -1;
}

vlong
archrelocvariant(Reloc *r, LSym *s, vlong t)
{
	uint32 o1;
	switch(r->variant & RV_TYPE_MASK) {
	default:
		diag("unexpected relocation variant %d", r->variant);

	case RV_NONE:
		return t;

	case RV_POWER_LO:
		if(r->variant & RV_CHECK_OVERFLOW) {
			// Whether to check for signed or unsigned
			// overflow depends on the instruction
			if(ctxt->arch->endian == BigEndian)
				o1 = be32(s->p + r->off - 2);
			else
				o1 = le32(s->p + r->off);
			switch(o1 >> 26) {
			case 24:	// ori
			case 26:	// xori
			case 28:	// andi
				if((t >> 16) != 0)
					goto overflow;
				break;
			default:
				if((int16)t != t)
					goto overflow;
				break;
			}
		}
		return (int16)t;

	case RV_POWER_HA:
		t += 0x8000;
		// Fallthrough
	case RV_POWER_HI:
		t >>= 16;
		if(r->variant & RV_CHECK_OVERFLOW) {
			// Whether to check for signed or unsigned
			// overflow depends on the instruction
			if(ctxt->arch->endian == BigEndian)
				o1 = be32(s->p + r->off - 2);
			else
				o1 = le32(s->p + r->off);
			switch(o1 >> 26) {
			case 25:	// oris
			case 27:	// xoris
			case 29:	// andis
				if((t >> 16) != 0)
					goto overflow;
				break;
			default:
				if((int16)t != t)
					goto overflow;
				break;
			}
		}
		return (int16)t;

	case RV_POWER_DS:
		if(ctxt->arch->endian == BigEndian)
			o1 = be16(s->p + r->off);
		else
			o1 = le16(s->p + r->off);
		if(t & 3)
			diag("relocation for %s+%d is not aligned: %lld", r->sym->name, r->off, t);
		if((r->variant & RV_CHECK_OVERFLOW) && (int16)t != t)
			goto overflow;
		return (o1 & 0x3) | (vlong)(int16)t;
	}

overflow:
	diag("relocation for %s+%d is too big: %lld", r->sym->name, r->off, t);
	return t;
}

static void
addpltsym(Link *ctxt, LSym *s)
{
	if(s->plt >= 0)
		return;

	adddynsym(ctxt, s);

	if(iself) {
		LSym *plt, *rela, *glink;
		Reloc *r;

		plt = linklookup(ctxt, ".plt", 0);
		rela = linklookup(ctxt, ".rela.plt", 0);
		if(plt->size == 0)
			elfsetupplt();

		// Create the glink resolver if necessary
		glink = ensureglinkresolver();

		// Write symbol resolver stub (just a branch to the
		// glink resolver stub)
		r = addrel(glink);
		r->sym = glink;
		r->off = glink->size;
		r->siz = 4;
		r->type = R_CALLPOWER;
		adduint32(ctxt, glink, 0x48000000); // b .glink

		// In the ppc64 ABI, the dynamic linker is responsible
		// for writing the entire PLT.  We just need to
		// reserve 8 bytes for each PLT entry and generate a
		// JMP_SLOT dynamic relocation for it.
		//
		// TODO(austin): ABI v1 is different
		s->plt = plt->size;
		plt->size += 8;

		addaddrplus(ctxt, rela, plt, s->plt);
		adduint64(ctxt, rela, ELF64_R_INFO(s->dynid, R_PPC64_JMP_SLOT));
		adduint64(ctxt, rela, 0);
	} else {
		diag("addpltsym: unsupported binary format");
	}
}

// Generate the glink resolver stub if necessary and return the .glink section
static LSym*
ensureglinkresolver(void)
{
	LSym *glink, *s;
	Reloc *r;

	glink = linklookup(ctxt, ".glink", 0);
	if(glink->size != 0)
		return glink;

	// This is essentially the resolver from the ppc64 ELF ABI.
	// At entry, r12 holds the address of the symbol resolver stub
	// for the target routine and the argument registers hold the
	// arguments for the target routine.
	//
	// This stub is PIC, so first get the PC of label 1 into r11.
	// Other things will be relative to this.
	adduint32(ctxt, glink, 0x7c0802a6); // mflr r0
	adduint32(ctxt, glink, 0x429f0005); // bcl 20,31,1f
	adduint32(ctxt, glink, 0x7d6802a6); // 1: mflr r11
	adduint32(ctxt, glink, 0x7c0803a6); // mtlf r0

	// Compute the .plt array index from the entry point address.
	// Because this is PIC, everything is relative to label 1b (in
	// r11):
	//   r0 = ((r12 - r11) - (res_0 - r11)) / 4 = (r12 - res_0) / 4
	adduint32(ctxt, glink, 0x3800ffd0); // li r0,-(res_0-1b)=-48
	adduint32(ctxt, glink, 0x7c006214); // add r0,r0,r12
	adduint32(ctxt, glink, 0x7c0b0050); // sub r0,r0,r11
	adduint32(ctxt, glink, 0x7800f082); // srdi r0,r0,2

	// r11 = address of the first byte of the PLT
	r = addrel(glink);
	r->off = glink->size;
	r->sym = linklookup(ctxt, ".plt", 0);
	r->siz = 8;
	r->type = R_ADDRPOWER;
	// addis r11,0,.plt@ha; addi r11,r11,.plt@l
	r->add = (0x3d600000ull << 32) | 0x396b0000;
	glink->size += 8;

	// Load r12 = dynamic resolver address and r11 = DSO
	// identifier from the first two doublewords of the PLT.
	adduint32(ctxt, glink, 0xe98b0000); // ld r12,0(r11)
	adduint32(ctxt, glink, 0xe96b0008); // ld r11,8(r11)

	// Jump to the dynamic resolver
	adduint32(ctxt, glink, 0x7d8903a6); // mtctr r12
	adduint32(ctxt, glink, 0x4e800420); // bctr

	// The symbol resolvers must immediately follow.
	//   res_0:

	// Add DT_PPC64_GLINK .dynamic entry, which points to 32 bytes
	// before the first symbol resolver stub.
	s = linklookup(ctxt, ".dynamic", 0);
	elfwritedynentsymplus(s, DT_PPC64_GLINK, glink, glink->size - 32);

	return glink;
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
		else
			adduint16(ctxt, d, 1);

		/* value */
		if(s->type == SDYNIMPORT)
			adduint64(ctxt, d, 0);
		else
			addaddr(ctxt, d, s);

		/* size of object */
		adduint64(ctxt, d, s->size);
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
	} else {
		diag("adddynlib: unsupported binary format");
	}
}

void
asmb(void)
{
	uint32 symo;
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
		LPUT(0x647);			/* magic */
		LPUT(segtext.filelen);			/* sizes */
		LPUT(segdata.filelen);
		LPUT(segdata.len - segdata.filelen);
		LPUT(symsize);			/* nsyms */
		LPUT(entryvalue());		/* va of entry */
		LPUT(0L);
		LPUT(lcsize);
		break;
	case Hlinux:
	case Hfreebsd:
	case Hnetbsd:
	case Hopenbsd:
	case Hnacl:
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

vlong
rnd(vlong v, int32 r)
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
