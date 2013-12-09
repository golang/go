// Inferno utils/5l/obj.c
// http://code.google.com/p/inferno-os/source/browse/utils/5l/obj.c
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

#include <u.h>
#include <libc.h>
#include <bio.h>
#include <link.h>
#include "../cmd/5l/5.out.h"

// TODO: remove duplicate chipzero, chipfloat

static void finish(Link*);

static int
chipzero(Link *ctxt, float64 e)
{
	// We use GOARM=7 to gate the use of VFPv3 vmov (imm) instructions.
	if(ctxt->goarm < 7 || e != 0)
		return -1;
	return 0;
}

static int
chipfloat(Link *ctxt, float64 e)
{
	int n;
	ulong h1;
	int32 l, h;
	uint64 ei;

	// We use GOARM=7 to gate the use of VFPv3 vmov (imm) instructions.
	if(ctxt->goarm < 7)
		goto no;

	memmove(&ei, &e, 8);
	l = (int32)ei;
	h = (int32)(ei>>32);

	if(l != 0 || (h&0xffff) != 0)
		goto no;
	h1 = h & 0x7fc00000;
	if(h1 != 0x40000000 && h1 != 0x3fc00000)
		goto no;
	n = 0;

	// sign bit (a)
	if(h & 0x80000000)
		n |= 1<<7;

	// exp sign bit (b)
	if(h1 == 0x3fc00000)
		n |= 1<<6;

	// rest of exp and mantissa (cd-efgh)
	n |= (h >> 16) & 0x3f;

//print("match %.8lux %.8lux %d\n", l, h, n);
	return n;

no:
	return -1;
}

static LSym*
zsym(char *pn, Biobuf *f, LSym *h[])
{	
	int o;
	
	o = BGETC(f);
	if(o == 0)
		return nil;
	if(o < 0 || o >= NSYM || h[o] == nil)
		mangle(pn);
	return h[o];
}

static void
zaddr(Link *ctxt, char *pn, Biobuf *f, Addr *a, LSym *h[], LSym **pgotype)
{
	int i, c;
	int32 l;
	LSym *s, *gotype;
	Auto *u;
	uint64 v;

	a->type = BGETC(f);
	a->reg = BGETC(f);
	c = BGETC(f);
	if(c < 0 || c > NSYM){
		print("sym out of range: %d\n", c);
		BPUTC(f, ALAST+1);
		return;
	}
	a->sym = h[c];
	a->name = BGETC(f);
	gotype = zsym(pn, f, h);
	if(pgotype)
		*pgotype = gotype;

	if((schar)a->reg < 0 || a->reg > NREG) {
		print("register out of range %d\n", a->reg);
		BPUTC(f, ALAST+1);
		return;	/*  force real diagnostic */
	}

	if(a->type == D_CONST || a->type == D_OCONST) {
		if(a->name == D_EXTERN || a->name == D_STATIC) {
			s = a->sym;
			if(s != nil && (s->type == STEXT || s->type == SCONST || s->type == SXREF)) {
				if(0 && !s->fnptr && s->name[0] != '.')
					print("%s used as function pointer\n", s->name);
				s->fnptr = 1;	// over the top cos of SXREF
			}
		}
	}

	switch(a->type) {
	default:
		print("unknown type %d\n", a->type);
		BPUTC(f, ALAST+1);
		return;	/*  force real diagnostic */

	case D_NONE:
	case D_REG:
	case D_FREG:
	case D_PSR:
	case D_FPCR:
		break;

	case D_REGREG:
	case D_REGREG2:
		a->offset = BGETC(f);
		break;

	case D_CONST2:
		a->offset2 = BGETLE4(f);	// fall through
	case D_BRANCH:
	case D_OREG:
	case D_CONST:
	case D_OCONST:
	case D_SHIFT:
		a->offset = BGETLE4(f);
		break;

	case D_SCONST:
		Bread(f, a->u.sval, NSNAME);
		break;

	case D_FCONST:
		v = (uint32)BGETLE4(f);
		v |= (uint64)BGETLE4(f)<<32;
		memmove(&a->u.dval, &v, 8);
		break;
	}
	s = a->sym;
	if(s == nil)
		return;
	i = a->name;
	if(i != D_AUTO && i != D_PARAM) {
		if(s && gotype)
			s->gotype = gotype;
		return;
	}

	l = a->offset;
	for(u=ctxt->curauto; u; u=u->link)
		if(u->asym == s)
		if(u->type == i) {
			if(u->aoffset > l)
				u->aoffset = l;
			if(gotype)
				u->gotype = gotype;
			return;
		}

	u = emallocz(sizeof(Auto));
	u->link = ctxt->curauto;
	ctxt->curauto = u;
	u->asym = s;
	u->aoffset = l;
	u->type = i;
	u->gotype = gotype;
}

void
nopout5(Prog *p)
{
	p->as = ANOP;
	p->from.type = D_NONE;
	p->to.type = D_NONE;
}

void
ldobj5(Link *ctxt, Biobuf *f, char *pkg, int64 len, char *pn)
{
	int32 ipc;
	Prog *p;
	LSym *h[NSYM], *s;
	int v, o, r, skip;
	uint32 sig;
	char *name;
	int ntext;
	int32 eof, autosize;
	char src[1024], *x, literal[64];
	Prog *lastp;
	LSym *fromgotype;

	lastp = nil;
	ntext = 0;
	eof = Boffset(f) + len;
	src[0] = 0;
	pn = estrdup(pn); // we keep it in LSym* references

newloop:
	memset(h, 0, sizeof(h));
	ctxt->version++;
	ctxt->histfrogp = 0;
	ipc = ctxt->pc;
	skip = 0;

loop:
	if(f->state == Bracteof || Boffset(f) >= eof)
		goto eof;
	o = BGETC(f);
	if(o == Beof)
		goto eof;

	if(o <= AXXX || o >= ALAST) {
		ctxt->diag("%s:#%lld: opcode out of range: %#ux", pn, Boffset(f), o);
		sysfatal("probably not a .5 file");
	}
	if(o == ANAME || o == ASIGNAME) {
		sig = 0;
		if(o == ASIGNAME)
			sig = BGETLE4(f);
		v = BGETC(f); /* type */
		o = BGETC(f); /* sym */
		r = 0;
		if(v == D_STATIC)
			r = ctxt->version;
		name = Brdline(f, '\0');
		if(name == nil) {
			if(Blinelen(f) > 0) {
				fprint(2, "%s: name too long\n", pn);
				sysfatal("invalid object file");
			}
			goto eof;
		}
		x = expandpkg(name, pkg);
		s = linklookup(ctxt, x, r);
		if(x != name)
			free(x);

		if(sig != 0){
			if(s->sig != 0 && s->sig != sig)
				ctxt->diag("incompatible type signatures %ux(%s) and %ux(%s) for %s", s->sig, s->file, sig, pn, s->name);
			s->sig = sig;
			s->file = pn;
		}

		if(ctxt->debugread)
			print("	ANAME	%s\n", s->name);
		if(o < 0 || o >= nelem(h)) {
			fprint(2, "%s: mangled input file\n", pn);
			sysfatal("invalid object");
		}
		h[o] = s;
		if((v == D_EXTERN || v == D_STATIC) && s->type == 0)
			s->type = SXREF;
		if(v == D_FILE) {
			if(s->type != SFILE) {
				ctxt->histgen++;
				s->type = SFILE;
				s->value = ctxt->histgen;
			}
			if(ctxt->histfrogp < LinkMaxHist) {
				ctxt->histfrog[ctxt->histfrogp] = s;
				ctxt->histfrogp++;
			} else
				collapsefrog(ctxt, s);
			ctxt->dwarfaddfrag(s->value, s->name);
		}
		goto loop;
	}

	p = emallocz(sizeof(Prog));
	p->as = o;
	p->scond = BGETC(f);
	p->reg = BGETC(f);
	p->lineno = BGETLE4(f);

	zaddr(ctxt, pn, f, &p->from, h, &fromgotype);
	zaddr(ctxt, pn, f, &p->to, h, nil);

	if(p->as != ATEXT && p->as != AGLOBL && p->reg > NREG)
		ctxt->diag("register out of range %A %d", p->as, p->reg);

	p->link = nil;
	p->pcond = nil;

	if(ctxt->debugread)
		print("%P\n", p);

	switch(o) {
	case AHISTORY:
		if(p->to.offset == -1) {
			addlib(ctxt, src, pn);
			ctxt->histfrogp = 0;
			goto loop;
		}
		if(src[0] == '\0')
			copyhistfrog(ctxt, src, sizeof src);
		addhist(ctxt, p->lineno, D_FILE);		/* 'z' */
		if(p->to.offset)
			addhist(ctxt, p->to.offset, D_FILE1);	/* 'Z' */
		savehist(ctxt, p->lineno, p->to.offset);
		ctxt->histfrogp = 0;
		goto loop;

	case AEND:
		finish(ctxt);
		if(Boffset(f) == eof)
			return;
		goto newloop;

	case AGLOBL:
		s = p->from.sym;
		if(s == nil) {
			ctxt->diag("GLOBL must have a name\n%P", p);
			sysfatal("mangled input");
		}
		if(s->type == 0 || s->type == SXREF) {
			s->type = SBSS;
			s->value = 0;
		}
		if(s->type != SBSS && s->type != SNOPTRBSS && !s->dupok) {
			ctxt->diag("redefinition: %s\n%P", s->name, p);
			s->type = SBSS;
			s->value = 0;
		}
		if(p->to.offset > s->size)
			s->size = p->to.offset;
		if(p->reg & DUPOK)
			s->dupok = 1;
		if(p->reg & RODATA)
			s->type = SRODATA;
		else if(p->reg & NOPTR)
			s->type = SNOPTRBSS;
		break;

	case ADATA:
		// Assume that AGLOBL comes after ADATA.
		// If we've seen an AGLOBL that said this sym was DUPOK,
		// ignore any more ADATA we see, which must be
		// redefinitions.
		s = p->from.sym;
		if(s->dupok) {
//			if(debug['v'])
//				Bprint(&bso, "skipping %s in %s: dupok\n", s->name, pn);
			goto loop;
		}
		if(s->file == nil)
			s->file = pn;
		else if(s->file != pn) {
			ctxt->diag("multiple initialization for %s: in both %s and %s", s->name, s->file, pn);
			sysfatal("mangled input");
		}
		savedata(ctxt, s, p, pn);
		free(p);
		break;

	case AGOK:
		ctxt->diag("unknown opcode\n%P", p);
		p->pc = ctxt->pc;
		ctxt->pc++;
		break;

	case ATYPE:
		if(skip)
			goto casedef;
		ctxt->pc++;
		goto loop;

	case ATEXT:
		if(ctxt->cursym != nil && ctxt->cursym->text)
			finish(ctxt);
		s = p->from.sym;
		if(s == nil) {
			ctxt->diag("TEXT must have a name\n%P", p);
			sysfatal("mangled input");
		}
		ctxt->cursym = s;
		if(s->type != 0 && s->type != SXREF && (p->reg & DUPOK)) {
			skip = 1;
			goto casedef;
		}
		if(ntext++ == 0 && s->type != 0 && s->type != SXREF) {
			/* redefinition, so file has probably been seen before */
			if(ctxt->debugvlog)
				Bprint(ctxt->bso, "skipping: %s: redefinition: %s", pn, s->name);
			return;
		}
		skip = 0;
		if(s->type != 0 && s->type != SXREF)
			ctxt->diag("redefinition: %s\n%P", s->name, p);
		if(ctxt->etextp)
			ctxt->etextp->next = s;
		else
			ctxt->textp = s;
		if(fromgotype) {
			if(s->gotype && s->gotype != fromgotype)
				ctxt->diag("%s: type mismatch for %s", pn, s->name);
			s->gotype = fromgotype;
		}
		ctxt->etextp = s;
		autosize = (p->to.offset+3L) & ~3L;
		p->to.offset = autosize;
		autosize += 4;
		s->type = STEXT;
		s->hist = gethist(ctxt);
		s->text = p;
		s->value = ctxt->pc;
		s->args = p->to.offset2;
		lastp = p;
		p->pc = ctxt->pc;
		ctxt->pc++;
		break;

	case ASUB:
		if(p->from.type == D_CONST)
		if(p->from.name == D_NONE)
		if(p->from.offset < 0) {
			p->from.offset = -p->from.offset;
			p->as = AADD;
		}
		goto casedef;

	case AADD:
		if(p->from.type == D_CONST)
		if(p->from.name == D_NONE)
		if(p->from.offset < 0) {
			p->from.offset = -p->from.offset;
			p->as = ASUB;
		}
		goto casedef;

	case AMOVWD:
	case AMOVWF:
	case AMOVDW:
	case AMOVFW:
	case AMOVFD:
	case AMOVDF:
	// case AMOVF:
	// case AMOVD:
	case ACMPF:
	case ACMPD:
	case AADDF:
	case AADDD:
	case ASUBF:
	case ASUBD:
	case AMULF:
	case AMULD:
	case ADIVF:
	case ADIVD:
		goto casedef;

	case AMOVF:
		if(skip)
			goto casedef;

		if(p->from.type == D_FCONST && chipfloat(ctxt, p->from.u.dval) < 0 &&
		   (chipzero(ctxt, p->from.u.dval) < 0 || (p->scond & C_SCOND) != C_SCOND_NONE)) {
			/* size sb 9 max */
			sprint(literal, "$%.17gf", (float32)p->from.u.dval);
			s = linklookup(ctxt, literal, 0);
			if(s->type == 0) {
				float32 f32;
				int32 i32;
				s->type = SRODATA;
				f32 = p->from.u.dval;
				memmove(&i32, &f32, 4);
				adduint32(ctxt, s, i32);
				s->reachable = 0;
			}
			p->from.type = D_OREG;
			p->from.sym = s;
			p->from.name = D_EXTERN;
			p->from.offset = 0;
		}
		goto casedef;

	case AMOVD:
		if(skip)
			goto casedef;

		if(p->from.type == D_FCONST && chipfloat(ctxt, p->from.u.dval) < 0 &&
		   (chipzero(ctxt, p->from.u.dval) < 0 || (p->scond & C_SCOND) != C_SCOND_NONE)) {
			/* size sb 18 max */
			sprint(literal, "$%.17g", p->from.u.dval);
			s = linklookup(ctxt, literal, 0);
			if(s->type == 0) {
				int64 i64;
				s->type = SRODATA;
				memmove(&i64, &p->from.u.dval, 8);
				adduint64(ctxt, s, i64);
				s->reachable = 0;
			}
			p->from.type = D_OREG;
			p->from.sym = s;
			p->from.name = D_EXTERN;
			p->from.offset = 0;
		}
		goto casedef;

	default:
	casedef:
		if(skip)
			nopout5(p);
		p->pc = ctxt->pc;
		ctxt->pc++;
		if(p->to.type == D_BRANCH)
			p->to.offset += ipc;
		if(lastp == nil) {
			if(p->as != ANOP)
				ctxt->diag("unexpected instruction: %P", p);
			break;
		}
		lastp->link = p;
		lastp = p;
		break;
	}
	goto loop;

eof:
	ctxt->diag("truncated object file: %s", pn);
}

static void
finish(Link *ctxt)
{
	LSym *s;
	
	histtoauto(ctxt);
	if(ctxt->cursym != nil && ctxt->cursym->text) {
		s = ctxt->cursym;
		s->autom = ctxt->curauto;
	//	mkfwd(s);
	//	linkpatch(ctxt, s);
	//	ctxt->arch->follow(ctxt, s);
	//	ctxt->arch->addstacksplit(ctxt, s);
	//	ctxt->arch->assemble(ctxt, s);
	//	linkpcln(ctxt, s);
	}

	ctxt->curauto = 0;
	ctxt->cursym = nil;
}

