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

// Data layout and relocation.

#include	"l.h"
#include	"../ld/lib.h"
#include	"../ld/elf.h"
#include	"../ld/macho.h"
#include	"../ld/pe.h"
#include	"../../runtime/mgc0.h"

void	dynreloc(void);

/*
 * divide-and-conquer list-link
 * sort of LSym* structures.
 * Used for the data block.
 */
int
datcmp(LSym *s1, LSym *s2)
{
	if(s1->type != s2->type)
		return (int)s1->type - (int)s2->type;
	if(s1->size != s2->size) {
		if(s1->size < s2->size)
			return -1;
		return +1;
	}
	return strcmp(s1->name, s2->name);
}

LSym*
listsort(LSym *l, int (*cmp)(LSym*, LSym*), int off)
{
	LSym *l1, *l2, *le;
	#define NEXT(l) (*(LSym**)((char*)(l)+off))

	if(l == 0 || NEXT(l) == 0)
		return l;

	l1 = l;
	l2 = l;
	for(;;) {
		l2 = NEXT(l2);
		if(l2 == 0)
			break;
		l2 = NEXT(l2);
		if(l2 == 0)
			break;
		l1 = NEXT(l1);
	}

	l2 = NEXT(l1);
	NEXT(l1) = 0;
	l1 = listsort(l, cmp, off);
	l2 = listsort(l2, cmp, off);

	/* set up lead element */
	if(cmp(l1, l2) < 0) {
		l = l1;
		l1 = NEXT(l1);
	} else {
		l = l2;
		l2 = NEXT(l2);
	}
	le = l;

	for(;;) {
		if(l1 == 0) {
			while(l2) {
				NEXT(le) = l2;
				le = l2;
				l2 = NEXT(l2);
			}
			NEXT(le) = 0;
			break;
		}
		if(l2 == 0) {
			while(l1) {
				NEXT(le) = l1;
				le = l1;
				l1 = NEXT(l1);
			}
			break;
		}
		if(cmp(l1, l2) < 0) {
			NEXT(le) = l1;
			le = l1;
			l1 = NEXT(l1);
		} else {
			NEXT(le) = l2;
			le = l2;
			l2 = NEXT(l2);
		}
	}
	NEXT(le) = 0;
	return l;
	
	#undef NEXT
}

void
relocsym(LSym *s)
{
	Reloc *r;
	LSym *rs;
	int32 i, off, siz, fl;
	vlong o;
	uchar *cast;

	ctxt->cursym = s;
	for(r=s->r; r<s->r+s->nr; r++) {
		r->done = 1;
		off = r->off;
		siz = r->siz;
		if(off < 0 || off+siz > s->np) {
			diag("%s: invalid relocation %d+%d not in [%d,%d)", s->name, off, siz, 0, s->np);
			continue;
		}
		if(r->sym != S && ((r->sym->type & (SMASK | SHIDDEN)) == 0 || (r->sym->type & SMASK) == SXREF)) {
			diag("%s: not defined", r->sym->name);
			continue;
		}
		if(r->type >= 256)
			continue;
		if(r->siz == 0) // informational relocation - no work to do
			continue;

		// Solaris needs the ability to reference dynimport symbols.
		if(HEADTYPE != Hsolaris && r->sym != S && r->sym->type == SDYNIMPORT)
			diag("unhandled relocation for %s (type %d rtype %d)", r->sym->name, r->sym->type, r->type);
		if(r->sym != S && r->sym->type != STLSBSS && !r->sym->reachable)
			diag("unreachable sym in relocation: %s %s", s->name, r->sym->name);

		// Android emulates runtime.tlsg as a regular variable.
		if (r->type == R_TLS && strcmp(goos, "android") == 0)
			r->type = R_ADDR;

		switch(r->type) {
		default:
			o = 0;
			if(archreloc(r, s, &o) < 0)
				diag("unknown reloc %d", r->type);
			break;
		case R_TLS:
			if(linkmode == LinkInternal && iself && thechar == '5') {
				// On ELF ARM, the thread pointer is 8 bytes before
				// the start of the thread-local data block, so add 8
				// to the actual TLS offset (r->sym->value).
				// This 8 seems to be a fundamental constant of
				// ELF on ARM (or maybe Glibc on ARM); it is not
				// related to the fact that our own TLS storage happens
				// to take up 8 bytes.
				o = 8 + r->sym->value;
				break;
			}
			r->done = 0;
			o = 0;
			if(thechar != '6')
				o = r->add;
			break;
		case R_TLS_LE:
			if(linkmode == LinkExternal && iself && HEADTYPE != Hopenbsd) {
				r->done = 0;
				r->sym = ctxt->tlsg;
				r->xsym = ctxt->tlsg;
				r->xadd = r->add;
				o = 0;
				if(thechar != '6')
					o = r->add;
				break;
			}
			o = ctxt->tlsoffset + r->add;
			break;

		case R_TLS_IE:
			if(linkmode == LinkExternal && iself && HEADTYPE != Hopenbsd) {
				r->done = 0;
				r->sym = ctxt->tlsg;
				r->xsym = ctxt->tlsg;
				r->xadd = r->add;
				o = 0;
				if(thechar != '6')
					o = r->add;
				break;
			}
			if(iself || ctxt->headtype == Hplan9)
				o = ctxt->tlsoffset + r->add;
			else if(ctxt->headtype == Hwindows)
				o = r->add;
			else
				sysfatal("unexpected R_TLS_IE relocation for %s", headstr(ctxt->headtype));
			break;
		case R_ADDR:
			if(linkmode == LinkExternal && r->sym->type != SCONST) {
				r->done = 0;

				// set up addend for eventual relocation via outer symbol.
				rs = r->sym;
				r->xadd = r->add;
				while(rs->outer != nil) {
					r->xadd += symaddr(rs) - symaddr(rs->outer);
					rs = rs->outer;
				}
				if(rs->type != SHOSTOBJ && rs->type != SDYNIMPORT && rs->sect == nil)
					diag("missing section for %s", rs->name);
				r->xsym = rs;

				o = r->xadd;
				if(iself) {
					if(thechar == '6')
						o = 0;
				} else if(HEADTYPE == Hdarwin) {
					if(rs->type != SHOSTOBJ)
						o += symaddr(rs);
				} else {
					diag("unhandled pcrel relocation for %s", headstring);
				}
				break;
			}
			o = symaddr(r->sym) + r->add;

			// On amd64, 4-byte offsets will be sign-extended, so it is impossible to
			// access more than 2GB of static data; fail at link time is better than
			// fail at runtime. See http://golang.org/issue/7980.
			// Instead of special casing only amd64, we treat this as an error on all
			// 64-bit architectures so as to be future-proof.
			if((int32)o < 0 && PtrSize > 4 && siz == 4) {
				diag("non-pc-relative relocation address is too big: %#llux", o);
				errorexit();
			}
			break;
		case R_CALL:
		case R_PCREL:
			// r->sym can be null when CALL $(constant) is transformed from absolute PC to relative PC call.
			if(linkmode == LinkExternal && r->sym && r->sym->type != SCONST && r->sym->sect != ctxt->cursym->sect) {
				r->done = 0;

				// set up addend for eventual relocation via outer symbol.
				rs = r->sym;
				r->xadd = r->add;
				while(rs->outer != nil) {
					r->xadd += symaddr(rs) - symaddr(rs->outer);
					rs = rs->outer;
				}
				r->xadd -= r->siz; // relative to address after the relocated chunk
				if(rs->type != SHOSTOBJ && rs->type != SDYNIMPORT && rs->sect == nil)
					diag("missing section for %s", rs->name);
				r->xsym = rs;

				o = r->xadd;
				if(iself) {
					if(thechar == '6')
						o = 0;
				} else if(HEADTYPE == Hdarwin) {
					if(r->type == R_CALL) {
						if(rs->type != SHOSTOBJ)
							o += symaddr(rs) - rs->sect->vaddr;
						o -= r->off; // relative to section offset, not symbol
					} else {
						o += r->siz;
					}
				} else {
					diag("unhandled pcrel relocation for %s", headstring);
				}
				break;
			}
			o = 0;
			if(r->sym)
				o += symaddr(r->sym);
			// NOTE: The (int32) cast on the next line works around a bug in Plan 9's 8c
			// compiler. The expression s->value + r->off + r->siz is int32 + int32 +
			// uchar, and Plan 9 8c incorrectly treats the expression as type uint32
			// instead of int32, causing incorrect values when sign extended for adding
			// to o. The bug only occurs on Plan 9, because this C program is compiled by
			// the standard host compiler (gcc on most other systems).
			o += r->add - (s->value + r->off + (int32)r->siz);
			break;
		case R_SIZE:
			o = r->sym->size + r->add;
			break;
		}
//print("relocate %s %#llux (%#llux+%#llux, size %d) => %s %#llux +%#llx [%llx]\n", s->name, (uvlong)(s->value+off), (uvlong)s->value, (uvlong)r->off, r->siz, r->sym ? r->sym->name : "<nil>", (uvlong)symaddr(r->sym), (vlong)r->add, (vlong)o);
		switch(siz) {
		default:
			ctxt->cursym = s;
			diag("bad reloc size %#ux for %s", siz, r->sym->name);
		case 1:
			// TODO(rsc): Remove.
			s->p[off] = (int8)o;
			break;
		case 4:
			if(r->type == R_PCREL || r->type == R_CALL) {
				if(o != (int32)o)
					diag("pc-relative relocation address is too big: %#llx", o);
			} else {
				if(o != (int32)o && o != (uint32)o)
					diag("non-pc-relative relocation address is too big: %#llux", o);
			}
			fl = o;
			cast = (uchar*)&fl;
			for(i=0; i<4; i++)
				s->p[off+i] = cast[inuxi4[i]];
			break;
		case 8:
			cast = (uchar*)&o;
			for(i=0; i<8; i++)
				s->p[off+i] = cast[inuxi8[i]];
			break;
		}
	}
}

void
reloc(void)
{
	LSym *s;

	if(debug['v'])
		Bprint(&bso, "%5.2f reloc\n", cputime());
	Bflush(&bso);

	for(s=ctxt->textp; s!=S; s=s->next)
		relocsym(s);
	for(s=datap; s!=S; s=s->next)
		relocsym(s);
}

void
dynrelocsym(LSym *s)
{
	Reloc *r;

	if(HEADTYPE == Hwindows) {
		LSym *rel, *targ;

		rel = linklookup(ctxt, ".rel", 0);
		if(s == rel)
			return;
		for(r=s->r; r<s->r+s->nr; r++) {
			targ = r->sym;
			if(targ == nil)
				continue;
			if(!targ->reachable)
				diag("internal inconsistency: dynamic symbol %s is not reachable.", targ->name);
			if(r->sym->plt == -2 && r->sym->got != -2) { // make dynimport JMP table for PE object files.
				targ->plt = rel->size;
				r->sym = rel;
				r->add = targ->plt;

				// jmp *addr
				if(thechar == '8') {
					adduint8(ctxt, rel, 0xff);
					adduint8(ctxt, rel, 0x25);
					addaddr(ctxt, rel, targ);
					adduint8(ctxt, rel, 0x90);
					adduint8(ctxt, rel, 0x90);
				} else {
					adduint8(ctxt, rel, 0xff);
					adduint8(ctxt, rel, 0x24);
					adduint8(ctxt, rel, 0x25);
					addaddrplus4(ctxt, rel, targ, 0);
					adduint8(ctxt, rel, 0x90);
				}
			} else if(r->sym->plt >= 0) {
				r->sym = rel;
				r->add = targ->plt;
			}
		}
		return;
	}

	for(r=s->r; r<s->r+s->nr; r++) {
		if(r->sym != S && r->sym->type == SDYNIMPORT || r->type >= 256) {
			if(r->sym != S && !r->sym->reachable)
				diag("internal inconsistency: dynamic symbol %s is not reachable.", r->sym->name);
			adddynrel(s, r);
		}
	}
}

void
dynreloc(void)
{
	LSym *s;

	// -d suppresses dynamic loader format, so we may as well not
	// compute these sections or mark their symbols as reachable.
	if(debug['d'] && HEADTYPE != Hwindows)
		return;
	if(debug['v'])
		Bprint(&bso, "%5.2f reloc\n", cputime());
	Bflush(&bso);

	for(s=ctxt->textp; s!=S; s=s->next)
		dynrelocsym(s);
	for(s=datap; s!=S; s=s->next)
		dynrelocsym(s);
	if(iself)
		elfdynhash();
}

static void
blk(LSym *start, int64 addr, int64 size)
{
	LSym *sym;
	int64 eaddr;
	uchar *p, *ep;

	for(sym = start; sym != nil; sym = sym->next)
		if(!(sym->type&SSUB) && sym->value >= addr)
			break;

	eaddr = addr+size;
	for(; sym != nil; sym = sym->next) {
		if(sym->type&SSUB)
			continue;
		if(sym->value >= eaddr)
			break;
		if(sym->value < addr) {
			diag("phase error: addr=%#llx but sym=%#llx type=%d", (vlong)addr, (vlong)sym->value, sym->type);
			errorexit();
		}
		ctxt->cursym = sym;
		for(; addr < sym->value; addr++)
			cput(0);
		p = sym->p;
		ep = p + sym->np;
		while(p < ep)
			cput(*p++);
		addr += sym->np;
		for(; addr < sym->value+sym->size; addr++)
			cput(0);
		if(addr != sym->value+sym->size) {
			diag("phase error: addr=%#llx value+size=%#llx", (vlong)addr, (vlong)sym->value+sym->size);
			errorexit();
		}
	}

	for(; addr < eaddr; addr++)
		cput(0);
	cflush();
}

void
codeblk(int64 addr, int64 size)
{
	LSym *sym;
	int64 eaddr, n;
	uchar *q;

	if(debug['a'])
		Bprint(&bso, "codeblk [%#x,%#x) at offset %#llx\n", addr, addr+size, cpos());

	blk(ctxt->textp, addr, size);

	/* again for printing */
	if(!debug['a'])
		return;

	for(sym = ctxt->textp; sym != nil; sym = sym->next) {
		if(!sym->reachable)
			continue;
		if(sym->value >= addr)
			break;
	}

	eaddr = addr + size;
	for(; sym != nil; sym = sym->next) {
		if(!sym->reachable)
			continue;
		if(sym->value >= eaddr)
			break;

		if(addr < sym->value) {
			Bprint(&bso, "%-20s %.8llux|", "_", (vlong)addr);
			for(; addr < sym->value; addr++)
				Bprint(&bso, " %.2ux", 0);
			Bprint(&bso, "\n");
		}

		Bprint(&bso, "%.6llux\t%-20s\n", (vlong)addr, sym->name);
		n = sym->size;
		q = sym->p;

		while(n >= 16) {
			Bprint(&bso, "%.6ux\t%-20.16I\n", addr, q);
			addr += 16;
			q += 16;
			n -= 16;
		}
		if(n > 0)
			Bprint(&bso, "%.6ux\t%-20.*I\n", addr, (int)n, q);
		addr += n;
	}

	if(addr < eaddr) {
		Bprint(&bso, "%-20s %.8llux|", "_", (vlong)addr);
		for(; addr < eaddr; addr++)
			Bprint(&bso, " %.2ux", 0);
	}
	Bflush(&bso);
}

void
datblk(int64 addr, int64 size)
{
	LSym *sym;
	int64 i, eaddr;
	uchar *p, *ep;
	char *typ, *rsname;
	Reloc *r;

	if(debug['a'])
		Bprint(&bso, "datblk [%#x,%#x) at offset %#llx\n", addr, addr+size, cpos());

	blk(datap, addr, size);

	/* again for printing */
	if(!debug['a'])
		return;

	for(sym = datap; sym != nil; sym = sym->next)
		if(sym->value >= addr)
			break;

	eaddr = addr + size;
	for(; sym != nil; sym = sym->next) {
		if(sym->value >= eaddr)
			break;
		if(addr < sym->value) {
			Bprint(&bso, "\t%.8ux| 00 ...\n", addr);
			addr = sym->value;
		}
		Bprint(&bso, "%s\n\t%.8ux|", sym->name, (uint)addr);
		p = sym->p;
		ep = p + sym->np;
		while(p < ep) {
			if(p > sym->p && (int)(p-sym->p)%16 == 0)
				Bprint(&bso, "\n\t%.8ux|", (uint)(addr+(p-sym->p)));
			Bprint(&bso, " %.2ux", *p++);
		}
		addr += sym->np;
		for(; addr < sym->value+sym->size; addr++)
			Bprint(&bso, " %.2ux", 0);
		Bprint(&bso, "\n");
		
		if(linkmode == LinkExternal) {
			for(i=0; i<sym->nr; i++) {
				r = &sym->r[i];
				rsname = "";
				if(r->sym)
					rsname = r->sym->name;
				typ = "?";
				switch(r->type) {
				case R_ADDR:
					typ = "addr";
					break;
				case R_PCREL:
					typ = "pcrel";
					break;
				case R_CALL:
					typ = "call";
					break;
				}
				Bprint(&bso, "\treloc %.8ux/%d %s %s+%#llx [%#llx]\n",
					(uint)(sym->value+r->off), r->siz, typ, rsname, (vlong)r->add, (vlong)(r->sym->value+r->add));
			}
		}				
	}

	if(addr < eaddr)
		Bprint(&bso, "\t%.8ux| 00 ...\n", (uint)addr);
	Bprint(&bso, "\t%.8ux|\n", (uint)eaddr);
}

void
strnput(char *s, int n)
{
	for(; n > 0 && *s; s++) {
		cput(*s);
		n--;
	}
	while(n > 0) {
		cput(0);
		n--;
	}
}

void
addstrdata(char *name, char *value)
{
	LSym *s, *sp;
	char *p;
	uchar reachable;

	p = smprint("%s.str", name);
	sp = linklookup(ctxt, p, 0);
	free(p);
	addstring(sp, value);
	sp->type = SRODATA;

	s = linklookup(ctxt, name, 0);
	s->size = 0;
	s->dupok = 1;
	reachable = s->reachable;
	addaddr(ctxt, s, sp);
	adduintxx(ctxt, s, strlen(value), PtrSize);

	// addstring, addaddr, etc., mark the symbols as reachable.
	// In this case that is not necessarily true, so stick to what
	// we know before entering this function.
	s->reachable = reachable;
	sp->reachable = reachable;
}

vlong
addstring(LSym *s, char *str)
{
	int n;
	int32 r;

	if(s->type == 0)
		s->type = SNOPTRDATA;
	s->reachable = 1;
	r = s->size;
	n = strlen(str)+1;
	if(strcmp(s->name, ".shstrtab") == 0)
		elfsetstring(str, r);
	symgrow(ctxt, s, r+n);
	memmove(s->p+r, str, n);
	s->size += n;
	return r;
}

void
dosymtype(void)
{
	LSym *s;

	for(s = ctxt->allsym; s != nil; s = s->allsym) {
		if(s->np > 0) {
			if(s->type == SBSS)
				s->type = SDATA;
			if(s->type == SNOPTRBSS)
				s->type = SNOPTRDATA;
		}
	}
}

static int32
symalign(LSym *s)
{
	int32 align;

	if(s->align != 0)
		return s->align;

	align = MaxAlign;
	while(align > s->size && align > 1)
		align >>= 1;
	if(align < s->align)
		align = s->align;
	return align;
}
	
static vlong
aligndatsize(vlong datsize, LSym *s)
{
	return rnd(datsize, symalign(s));
}

// maxalign returns the maximum required alignment for
// the list of symbols s; the list stops when s->type exceeds type.
static int32
maxalign(LSym *s, int type)
{
	int32 align, max;
	
	max = 0;
	for(; s != S && s->type <= type; s = s->next) {
		align = symalign(s);
		if(max < align)
			max = align;
	}
	return max;
}

// Helper object for building GC type programs.
typedef struct ProgGen ProgGen;
struct ProgGen
{
	LSym*	s;
	int32	datasize;
	uint8	data[256/PointersPerByte];
	vlong	pos;
};

static void
proggeninit(ProgGen *g, LSym *s)
{
	g->s = s;
	g->datasize = 0;
	g->pos = 0;
	memset(g->data, 0, sizeof(g->data));
}

static void
proggenemit(ProgGen *g, uint8 v)
{
	adduint8(ctxt, g->s, v);
}

// Writes insData block from g->data.
static void
proggendataflush(ProgGen *g)
{
	int32 i, s;

	if(g->datasize == 0)
		return;
	proggenemit(g, insData);
	proggenemit(g, g->datasize);
	s = (g->datasize + PointersPerByte - 1)/PointersPerByte;
	for(i = 0; i < s; i++)
		proggenemit(g, g->data[i]);
	g->datasize = 0;
	memset(g->data, 0, sizeof(g->data));
}

static void
proggendata(ProgGen *g, uint8 d)
{
	g->data[g->datasize/PointersPerByte] |= d << ((g->datasize%PointersPerByte)*BitsPerPointer);
	g->datasize++;
	if(g->datasize == 255)
		proggendataflush(g);
}

// Skip v bytes due to alignment, etc.
static void
proggenskip(ProgGen *g, vlong off, vlong v)
{
	vlong i;

	for(i = off; i < off+v; i++) {
		if((i%PtrSize) == 0)
			proggendata(g, BitsScalar);
	}
}

// Emit insArray instruction.
static void
proggenarray(ProgGen *g, vlong len)
{
	int32 i;

	proggendataflush(g);
	proggenemit(g, insArray);
	for(i = 0; i < PtrSize; i++, len >>= 8)
		proggenemit(g, len);
}

static void
proggenarrayend(ProgGen *g)
{
	proggendataflush(g);
	proggenemit(g, insArrayEnd);
}

static void
proggenfini(ProgGen *g, vlong size)
{
	proggenskip(g, g->pos, size - g->pos);
	proggendataflush(g);
	proggenemit(g, insEnd);
}


// This function generates GC pointer info for global variables.
static void
proggenaddsym(ProgGen *g, LSym *s)
{
	LSym *gcprog;
	uint8 *mask;
	vlong i, size;

	if(s->size == 0)
		return;

	// Skip alignment hole from the previous symbol.
	proggenskip(g, g->pos, s->value - g->pos);
	g->pos += s->value - g->pos;

	// The test for names beginning with . here is meant
	// to keep .dynamic and .dynsym from turning up as
	// conservative symbols. They should be marked SELFSECT
	// and not SDATA, but sometimes that doesn't happen.
	// Leave debugging the SDATA issue for the Go rewrite.

	if(s->gotype == nil && s->size >= PtrSize && s->name[0] != '.') {
		// conservative scan
		diag("missing Go type information for global symbol: %s size %d", s->name, (int)s->size);
		if((s->size%PtrSize) || (g->pos%PtrSize))
			diag("proggenaddsym: unaligned conservative symbol %s: size=%lld pos=%lld",
				s->name, s->size, g->pos);
		size = (s->size+PtrSize-1)/PtrSize*PtrSize;
		if(size < 32*PtrSize) {
			// Emit small symbols as data.
			for(i = 0; i < size/PtrSize; i++)
				proggendata(g, BitsPointer);
		} else {
			// Emit large symbols as array.
			proggenarray(g, size/PtrSize);
			proggendata(g, BitsPointer);
			proggenarrayend(g);
		}
		g->pos = s->value + size;
	} else if(s->gotype == nil || decodetype_noptr(s->gotype) || s->size < PtrSize || s->name[0] == '.') {
		// no scan
		if(s->size < 32*PtrSize) {
			// Emit small symbols as data.
			// This case also handles unaligned and tiny symbols, so tread carefully.
			for(i = s->value; i < s->value+s->size; i++) {
				if((i%PtrSize) == 0)
					proggendata(g, BitsScalar);
			}
		} else {
			// Emit large symbols as array.
			if((s->size%PtrSize) || (g->pos%PtrSize))
				diag("proggenaddsym: unaligned noscan symbol %s: size=%lld pos=%lld",
					s->name, s->size, g->pos);
			proggenarray(g, s->size/PtrSize);
			proggendata(g, BitsScalar);
			proggenarrayend(g);
		}
		g->pos = s->value + s->size;
	} else if(decodetype_usegcprog(s->gotype)) {
		// gc program, copy directly
		proggendataflush(g);
		gcprog = decodetype_gcprog(s->gotype);
		size = decodetype_size(s->gotype);
		if((size%PtrSize) || (g->pos%PtrSize))
			diag("proggenaddsym: unaligned gcprog symbol %s: size=%lld pos=%lld",
				s->name, s->size, g->pos);
		for(i = 0; i < gcprog->np-1; i++)
			proggenemit(g, gcprog->p[i]);
		g->pos = s->value + size;
	} else {
		// gc mask, it's small so emit as data
		mask = decodetype_gcmask(s->gotype);
		size = decodetype_size(s->gotype);
		if((size%PtrSize) || (g->pos%PtrSize))
			diag("proggenaddsym: unaligned gcmask symbol %s: size=%lld pos=%lld",
				s->name, s->size, g->pos);
		for(i = 0; i < size; i += PtrSize)
			proggendata(g, (mask[i/PtrSize/2]>>((i/PtrSize%2)*4+2))&BitsMask);
		g->pos = s->value + size;
	}
}

void
growdatsize(vlong *datsizep, LSym *s)
{
	vlong datsize;
	
	datsize = *datsizep;
	if(s->size < 0)
		diag("negative size (datsize = %lld, s->size = %lld)", datsize, s->size);
	if(datsize + s->size < datsize)
		diag("symbol too large (datsize = %lld, s->size = %lld)", datsize, s->size);
	*datsizep = datsize + s->size;
}

void
dodata(void)
{
	int32 n;
	vlong datsize;
	Section *sect;
	Segment *segro;
	LSym *s, *last, **l;
	LSym *gcdata, *gcbss;
	ProgGen gen;

	if(debug['v'])
		Bprint(&bso, "%5.2f dodata\n", cputime());
	Bflush(&bso);

	last = nil;
	datap = nil;

	for(s=ctxt->allsym; s!=S; s=s->allsym) {
		if(!s->reachable || s->special)
			continue;
		if(STEXT < s->type && s->type < SXREF) {
			if(s->onlist)
				sysfatal("symbol %s listed multiple times", s->name);
			s->onlist = 1;
			if(last == nil)
				datap = s;
			else
				last->next = s;
			s->next = nil;
			last = s;
		}
	}

	for(s = datap; s != nil; s = s->next) {
		if(s->np > s->size)
			diag("%s: initialize bounds (%lld < %d)",
				s->name, (vlong)s->size, s->np);
	}


	/*
	 * now that we have the datap list, but before we start
	 * to assign addresses, record all the necessary
	 * dynamic relocations.  these will grow the relocation
	 * symbol, which is itself data.
	 *
	 * on darwin, we need the symbol table numbers for dynreloc.
	 */
	if(HEADTYPE == Hdarwin)
		machosymorder();
	dynreloc();

	/* some symbols may no longer belong in datap (Mach-O) */
	for(l=&datap; (s=*l) != nil; ) {
		if(s->type <= STEXT || SXREF <= s->type)
			*l = s->next;
		else
			l = &s->next;
	}
	*l = nil;

	datap = listsort(datap, datcmp, offsetof(LSym, next));

	/*
	 * allocate sections.  list is sorted by type,
	 * so we can just walk it for each piece we want to emit.
	 * segdata is processed before segtext, because we need
	 * to see all symbols in the .data and .bss sections in order
	 * to generate garbage collection information.
	 */

	/* begin segdata */

	/* skip symbols belonging to segtext */
	s = datap;
	for(; s != nil && s->type < SELFSECT; s = s->next)
		;

	/* writable ELF sections */
	datsize = 0;
	for(; s != nil && s->type < SNOPTRDATA; s = s->next) {
		sect = addsection(&segdata, s->name, 06);
		sect->align = symalign(s);
		datsize = rnd(datsize, sect->align);
		sect->vaddr = datsize;
		s->sect = sect;
		s->type = SDATA;
		s->value = datsize - sect->vaddr;
		growdatsize(&datsize, s);
		sect->len = datsize - sect->vaddr;
	}

	/* pointer-free data */
	sect = addsection(&segdata, ".noptrdata", 06);
	sect->align = maxalign(s, SINITARR-1);
	datsize = rnd(datsize, sect->align);
	sect->vaddr = datsize;
	linklookup(ctxt, "runtime.noptrdata", 0)->sect = sect;
	linklookup(ctxt, "runtime.enoptrdata", 0)->sect = sect;
	for(; s != nil && s->type < SINITARR; s = s->next) {
		datsize = aligndatsize(datsize, s);
		s->sect = sect;
		s->type = SDATA;
		s->value = datsize - sect->vaddr;
		growdatsize(&datsize, s);
	}
	sect->len = datsize - sect->vaddr;

	/* shared library initializer */
	if(flag_shared) {
		sect = addsection(&segdata, ".init_array", 06);
		sect->align = maxalign(s, SINITARR);
		datsize = rnd(datsize, sect->align);
		sect->vaddr = datsize;
		for(; s != nil && s->type == SINITARR; s = s->next) {
			datsize = aligndatsize(datsize, s);
			s->sect = sect;
			s->value = datsize - sect->vaddr;
			growdatsize(&datsize, s);
		}
		sect->len = datsize - sect->vaddr;
	}

	/* data */
	sect = addsection(&segdata, ".data", 06);
	sect->align = maxalign(s, SBSS-1);
	datsize = rnd(datsize, sect->align);
	sect->vaddr = datsize;
	linklookup(ctxt, "runtime.data", 0)->sect = sect;
	linklookup(ctxt, "runtime.edata", 0)->sect = sect;
	gcdata = linklookup(ctxt, "runtime.gcdata", 0);
	proggeninit(&gen, gcdata);
	for(; s != nil && s->type < SBSS; s = s->next) {
		if(s->type == SINITARR) {
			ctxt->cursym = s;
			diag("unexpected symbol type %d", s->type);
		}
		s->sect = sect;
		s->type = SDATA;
		datsize = aligndatsize(datsize, s);
		s->value = datsize - sect->vaddr;
		proggenaddsym(&gen, s);  // gc
		growdatsize(&datsize, s);
	}
	sect->len = datsize - sect->vaddr;
	proggenfini(&gen, sect->len);  // gc

	/* bss */
	sect = addsection(&segdata, ".bss", 06);
	sect->align = maxalign(s, SNOPTRBSS-1);
	datsize = rnd(datsize, sect->align);
	sect->vaddr = datsize;
	linklookup(ctxt, "runtime.bss", 0)->sect = sect;
	linklookup(ctxt, "runtime.ebss", 0)->sect = sect;
	gcbss = linklookup(ctxt, "runtime.gcbss", 0);
	proggeninit(&gen, gcbss);
	for(; s != nil && s->type < SNOPTRBSS; s = s->next) {
		s->sect = sect;
		datsize = aligndatsize(datsize, s);
		s->value = datsize - sect->vaddr;
		proggenaddsym(&gen, s);  // gc
		growdatsize(&datsize, s);
	}
	sect->len = datsize - sect->vaddr;
	proggenfini(&gen, sect->len);  // gc

	/* pointer-free bss */
	sect = addsection(&segdata, ".noptrbss", 06);
	sect->align = maxalign(s, SNOPTRBSS);
	datsize = rnd(datsize, sect->align);
	sect->vaddr = datsize;
	linklookup(ctxt, "runtime.noptrbss", 0)->sect = sect;
	linklookup(ctxt, "runtime.enoptrbss", 0)->sect = sect;
	for(; s != nil && s->type == SNOPTRBSS; s = s->next) {
		datsize = aligndatsize(datsize, s);
		s->sect = sect;
		s->value = datsize - sect->vaddr;
		growdatsize(&datsize, s);
	}
	sect->len = datsize - sect->vaddr;
	linklookup(ctxt, "runtime.end", 0)->sect = sect;

	// 6g uses 4-byte relocation offsets, so the entire segment must fit in 32 bits.
	if(datsize != (uint32)datsize) {
		diag("data or bss segment too large");
	}
	
	if(iself && linkmode == LinkExternal && s != nil && s->type == STLSBSS && HEADTYPE != Hopenbsd) {
		sect = addsection(&segdata, ".tbss", 06);
		sect->align = PtrSize;
		sect->vaddr = 0;
		datsize = 0;
		for(; s != nil && s->type == STLSBSS; s = s->next) {
			datsize = aligndatsize(datsize, s);
			s->sect = sect;
			s->value = datsize - sect->vaddr;
			growdatsize(&datsize, s);
		}
		sect->len = datsize;
	} else {
		// Might be internal linking but still using cgo.
		// In that case, the only possible STLSBSS symbol is runtime.tlsg.
		// Give it offset 0, because it's the only thing here.
		if(s != nil && s->type == STLSBSS && strcmp(s->name, "runtime.tlsg") == 0) {
			s->value = 0;
			s = s->next;
		}
	}
	
	if(s != nil) {
		ctxt->cursym = nil;
		diag("unexpected symbol type %d for %s", s->type, s->name);
	}

	/*
	 * We finished data, begin read-only data.
	 * Not all systems support a separate read-only non-executable data section.
	 * ELF systems do.
	 * OS X and Plan 9 do not.
	 * Windows PE may, but if so we have not implemented it.
	 * And if we're using external linking mode, the point is moot,
	 * since it's not our decision; that code expects the sections in
	 * segtext.
	 */
	if(iself && linkmode == LinkInternal)
		segro = &segrodata;
	else
		segro = &segtext;

	s = datap;
	
	datsize = 0;
	
	/* read-only executable ELF, Mach-O sections */
	for(; s != nil && s->type < STYPE; s = s->next) {
		sect = addsection(&segtext, s->name, 04);
		sect->align = symalign(s);
		datsize = rnd(datsize, sect->align);
		sect->vaddr = datsize;
		s->sect = sect;
		s->type = SRODATA;
		s->value = datsize - sect->vaddr;
		growdatsize(&datsize, s);
		sect->len = datsize - sect->vaddr;
	}

	/* read-only data */
	sect = addsection(segro, ".rodata", 04);
	sect->align = maxalign(s, STYPELINK-1);
	datsize = rnd(datsize, sect->align);
	sect->vaddr = 0;
	linklookup(ctxt, "runtime.rodata", 0)->sect = sect;
	linklookup(ctxt, "runtime.erodata", 0)->sect = sect;
	for(; s != nil && s->type < STYPELINK; s = s->next) {
		datsize = aligndatsize(datsize, s);
		s->sect = sect;
		s->type = SRODATA;
		s->value = datsize - sect->vaddr;
		growdatsize(&datsize, s);
	}
	sect->len = datsize - sect->vaddr;

	/* typelink */
	sect = addsection(segro, ".typelink", 04);
	sect->align = maxalign(s, STYPELINK);
	datsize = rnd(datsize, sect->align);
	sect->vaddr = datsize;
	linklookup(ctxt, "runtime.typelink", 0)->sect = sect;
	linklookup(ctxt, "runtime.etypelink", 0)->sect = sect;
	for(; s != nil && s->type == STYPELINK; s = s->next) {
		datsize = aligndatsize(datsize, s);
		s->sect = sect;
		s->type = SRODATA;
		s->value = datsize - sect->vaddr;
		growdatsize(&datsize, s);
	}
	sect->len = datsize - sect->vaddr;

	/* gosymtab */
	sect = addsection(segro, ".gosymtab", 04);
	sect->align = maxalign(s, SPCLNTAB-1);
	datsize = rnd(datsize, sect->align);
	sect->vaddr = datsize;
	linklookup(ctxt, "runtime.symtab", 0)->sect = sect;
	linklookup(ctxt, "runtime.esymtab", 0)->sect = sect;
	for(; s != nil && s->type < SPCLNTAB; s = s->next) {
		datsize = aligndatsize(datsize, s);
		s->sect = sect;
		s->type = SRODATA;
		s->value = datsize - sect->vaddr;
		growdatsize(&datsize, s);
	}
	sect->len = datsize - sect->vaddr;

	/* gopclntab */
	sect = addsection(segro, ".gopclntab", 04);
	sect->align = maxalign(s, SELFROSECT-1);
	datsize = rnd(datsize, sect->align);
	sect->vaddr = datsize;
	linklookup(ctxt, "runtime.pclntab", 0)->sect = sect;
	linklookup(ctxt, "runtime.epclntab", 0)->sect = sect;
	for(; s != nil && s->type < SELFROSECT; s = s->next) {
		datsize = aligndatsize(datsize, s);
		s->sect = sect;
		s->type = SRODATA;
		s->value = datsize - sect->vaddr;
		growdatsize(&datsize, s);
	}
	sect->len = datsize - sect->vaddr;

	/* read-only ELF, Mach-O sections */
	for(; s != nil && s->type < SELFSECT; s = s->next) {
		sect = addsection(segro, s->name, 04);
		sect->align = symalign(s);
		datsize = rnd(datsize, sect->align);
		sect->vaddr = datsize;
		s->sect = sect;
		s->type = SRODATA;
		s->value = datsize - sect->vaddr;
		growdatsize(&datsize, s);
		sect->len = datsize - sect->vaddr;
	}

	// 6g uses 4-byte relocation offsets, so the entire segment must fit in 32 bits.
	if(datsize != (uint32)datsize) {
		diag("read-only data segment too large");
	}
	
	/* number the sections */
	n = 1;
	for(sect = segtext.sect; sect != nil; sect = sect->next)
		sect->extnum = n++;
	for(sect = segrodata.sect; sect != nil; sect = sect->next)
		sect->extnum = n++;
	for(sect = segdata.sect; sect != nil; sect = sect->next)
		sect->extnum = n++;
}

// assign addresses to text
void
textaddress(void)
{
	uvlong va;
	Section *sect;
	LSym *sym, *sub;

	addsection(&segtext, ".text", 05);

	// Assign PCs in text segment.
	// Could parallelize, by assigning to text
	// and then letting threads copy down, but probably not worth it.
	sect = segtext.sect;
	sect->align = funcalign;
	linklookup(ctxt, "runtime.text", 0)->sect = sect;
	linklookup(ctxt, "runtime.etext", 0)->sect = sect;
	va = INITTEXT;
	sect->vaddr = va;
	for(sym = ctxt->textp; sym != nil; sym = sym->next) {
		sym->sect = sect;
		if(sym->type & SSUB)
			continue;
		if(sym->align != 0)
			va = rnd(va, sym->align);
		else
			va = rnd(va, funcalign);
		sym->value = 0;
		for(sub = sym; sub != S; sub = sub->sub)
			sub->value += va;
		if(sym->size == 0 && sym->sub != S)
			ctxt->cursym = sym;
		va += sym->size;
	}
	sect->len = va - sect->vaddr;
}

// assign addresses
void
address(void)
{
	Section *s, *text, *data, *rodata, *symtab, *pclntab, *noptr, *bss, *noptrbss;
	Section *typelink;
	LSym *sym, *sub;
	uvlong va;
	vlong vlen;

	va = INITTEXT;
	segtext.rwx = 05;
	segtext.vaddr = va;
	segtext.fileoff = HEADR;
	for(s=segtext.sect; s != nil; s=s->next) {
		va = rnd(va, s->align);
		s->vaddr = va;
		va += s->len;
	}
	segtext.len = va - INITTEXT;
	segtext.filelen = segtext.len;
	if(HEADTYPE == Hnacl)
		va += 32; // room for the "halt sled"

	if(segrodata.sect != nil) {
		// align to page boundary so as not to mix
		// rodata and executable text.
		va = rnd(va, INITRND);

		segrodata.rwx = 04;
		segrodata.vaddr = va;
		segrodata.fileoff = va - segtext.vaddr + segtext.fileoff;
		segrodata.filelen = 0;
		for(s=segrodata.sect; s != nil; s=s->next) {
			va = rnd(va, s->align);
			s->vaddr = va;
			va += s->len;
		}
		segrodata.len = va - segrodata.vaddr;
		segrodata.filelen = segrodata.len;
	}

	va = rnd(va, INITRND);
	segdata.rwx = 06;
	segdata.vaddr = va;
	segdata.fileoff = va - segtext.vaddr + segtext.fileoff;
	segdata.filelen = 0;
	if(HEADTYPE == Hwindows)
		segdata.fileoff = segtext.fileoff + rnd(segtext.len, PEFILEALIGN);
	if(HEADTYPE == Hplan9)
		segdata.fileoff = segtext.fileoff + segtext.filelen;
	data = nil;
	noptr = nil;
	bss = nil;
	noptrbss = nil;
	for(s=segdata.sect; s != nil; s=s->next) {
		vlen = s->len;
		if(s->next)
			vlen = s->next->vaddr - s->vaddr;
		s->vaddr = va;
		va += vlen;
		segdata.len = va - segdata.vaddr;
		if(strcmp(s->name, ".data") == 0)
			data = s;
		if(strcmp(s->name, ".noptrdata") == 0)
			noptr = s;
		if(strcmp(s->name, ".bss") == 0)
			bss = s;
		if(strcmp(s->name, ".noptrbss") == 0)
			noptrbss = s;
	}
	segdata.filelen = bss->vaddr - segdata.vaddr;

	text = segtext.sect;
	if(segrodata.sect)
		rodata = segrodata.sect;
	else
		rodata = text->next;
	typelink = rodata->next;
	symtab = typelink->next;
	pclntab = symtab->next;

	for(sym = datap; sym != nil; sym = sym->next) {
		ctxt->cursym = sym;
		if(sym->sect != nil)
			sym->value += sym->sect->vaddr;
		for(sub = sym->sub; sub != nil; sub = sub->sub)
			sub->value += sym->value;
	}

	xdefine("runtime.text", STEXT, text->vaddr);
	xdefine("runtime.etext", STEXT, text->vaddr + text->len);
	xdefine("runtime.rodata", SRODATA, rodata->vaddr);
	xdefine("runtime.erodata", SRODATA, rodata->vaddr + rodata->len);
	xdefine("runtime.typelink", SRODATA, typelink->vaddr);
	xdefine("runtime.etypelink", SRODATA, typelink->vaddr + typelink->len);

	sym = linklookup(ctxt, "runtime.gcdata", 0);
	xdefine("runtime.egcdata", SRODATA, symaddr(sym) + sym->size);
	linklookup(ctxt, "runtime.egcdata", 0)->sect = sym->sect;

	sym = linklookup(ctxt, "runtime.gcbss", 0);
	xdefine("runtime.egcbss", SRODATA, symaddr(sym) + sym->size);
	linklookup(ctxt, "runtime.egcbss", 0)->sect = sym->sect;

	xdefine("runtime.symtab", SRODATA, symtab->vaddr);
	xdefine("runtime.esymtab", SRODATA, symtab->vaddr + symtab->len);
	xdefine("runtime.pclntab", SRODATA, pclntab->vaddr);
	xdefine("runtime.epclntab", SRODATA, pclntab->vaddr + pclntab->len);
	xdefine("runtime.noptrdata", SNOPTRDATA, noptr->vaddr);
	xdefine("runtime.enoptrdata", SNOPTRDATA, noptr->vaddr + noptr->len);
	xdefine("runtime.bss", SBSS, bss->vaddr);
	xdefine("runtime.ebss", SBSS, bss->vaddr + bss->len);
	xdefine("runtime.data", SDATA, data->vaddr);
	xdefine("runtime.edata", SDATA, data->vaddr + data->len);
	xdefine("runtime.noptrbss", SNOPTRBSS, noptrbss->vaddr);
	xdefine("runtime.enoptrbss", SNOPTRBSS, noptrbss->vaddr + noptrbss->len);
	xdefine("runtime.end", SBSS, segdata.vaddr + segdata.len);
}
