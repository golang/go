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

void
adddynrela(LSym *rel, LSym *s, Reloc *r)
{
	// TODO(minux)
	USED(rel); USED(s); USED(r);
}

void
adddynrel(LSym *s, Reloc *r)
{
	LSym *targ;

	// TODO(minux)

	targ = r->sym;
	ctxt->cursym = s;
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
	// TODO(minux)
	return;
}

int
machoreloc1(Reloc *r, vlong sectoff)
{
	USED(r);
	USED(sectoff);

	return -1;
}


int
archreloc(Reloc *r, LSym *s, vlong *val)
{
	uint32 o1, o2;
	int32 t;

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
			ctxt->diag("relocation for %s is not aligned: %lld", s->name, t);
		if(t << 6 >> 6 != t)
			ctxt->diag("relocation for %s is too big: %lld", s->name, t);

		*val = (o1 & 0xfc000003U) | (t & ~0xfc000003U);
		return 0;
	}
	return -1;
}

void
adddynsym(Link *ctxt, LSym *s)
{
	USED(ctxt); USED(s);
	// TODO(minux)
	return;
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
