// Inferno utils/8l/obj.c
// http://code.google.com/p/inferno-os/source/browse/utils/8l/obj.c
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
#include "../cmd/8l/8.out.h"

static LSym*
zsym(char *pn, Biobuf *f, LSym *h[])
{	
	int o;
	
	o = BGETC(f);
	if(o < 0 || o >= NSYM || h[o] == nil)
		mangle(pn);
	return h[o];
}

static void finish(Link*);

static void
zaddr(Link *ctxt, char *pn, Biobuf *f, Addr *a, LSym *h[], LSym **pgotype)
{
	int t;
	int32 l;
	LSym *s, *gotype;
	Auto *u;
	uint64 v;

	t = BGETC(f);
	a->index = D_NONE;
	a->scale = 0;
	if(t & T_INDEX) {
		a->index = BGETC(f);
		a->scale = BGETC(f);
	}
	a->type = D_NONE;
	a->offset = 0;
	if(t & T_OFFSET)
		a->offset = BGETLE4(f);
	a->offset2 = 0;
	if(t & T_OFFSET2) {
		a->offset2 = BGETLE4(f);
		a->type = D_CONST2;
	}
	a->sym = nil;
	if(t & T_SYM)
		a->sym = zsym(pn, f, h);
	if(t & T_FCONST) {
		v = (uint32)BGETLE4(f);
		v |= (uint64)BGETLE4(f)<<32;
		memmove(&a->u.dval, &v, 8);
		a->type = D_FCONST;
	} else
	if(t & T_SCONST) {
		Bread(f, a->u.sval, NSNAME);
		a->type = D_SCONST;
	}
	if(t & T_TYPE)
		a->type = BGETC(f);
	gotype = nil;
	if(t & T_GOTYPE)
		gotype = zsym(pn, f, h);
	if(pgotype)
		*pgotype = gotype;
	t = a->type;
	if(t == D_INDIR+D_GS)
		a->offset += ctxt->tlsoffset;

	s = a->sym;
	if(s == nil)
		return;
	if(t != D_AUTO && t != D_PARAM) {
		if(gotype)
			s->gotype = gotype;
		return;
	}
	l = a->offset;
	for(u=ctxt->curauto; u; u=u->link) {
		if(u->asym == s)
		if(u->type == t) {
			if(u->aoffset > l)
				u->aoffset = l;
			if(gotype)
				u->gotype = gotype;
			return;
		}
	}

	u = emallocz(sizeof(*u));
	u->link = ctxt->curauto;
	ctxt->curauto = u;
	u->asym = s;
	u->aoffset = l;
	u->type = t;
	u->gotype = gotype;
}

void
nopout8(Prog *p)
{
	p->as = ANOP;
	p->from.type = D_NONE;
	p->to.type = D_NONE;
}

void
ldobj8(Link *ctxt, Biobuf *f, char *pkg, int64 len, char *pn)
{
	int32 ipc;
	Prog *p;
	int v, o, r, skip;
	LSym *h[NSYM], *s;
	uint32 sig;
	int ntext;
	int32 eof;
	char *name, *x;
	char src[1024], literal[64];
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
	o |= BGETC(f) << 8;
	if(o <= AXXX || o >= ALAST) {
		if(o < 0)
			goto eof;
		ctxt->diag("%s:#%lld: opcode out of range: %#ux", pn, Boffset(f), o);
		print("	probably not a .%c file\n", ctxt->thechar);
		sysfatal("invalid file");
	}

	if(o == ANAME || o == ASIGNAME) {
		sig = 0;
		if(o == ASIGNAME)
			sig = BGETLE4(f);
		USED(sig);

		v = BGETC(f);	/* type */
		o = BGETC(f);	/* sym */
		r = 0;
		if(v == D_STATIC)
			r = ctxt->version;
		name = Brdline(f, '\0');
		if(name == nil) {
			if(Blinelen(f) > 0)
				sysfatal("%s: name too long", pn);
			goto eof;
		}
		x = expandpkg(name, pkg);
		s = linklookup(ctxt, x, r);
		if(x != name)
			free(x);

		if(ctxt->debugread)
			print("	ANAME	%s\n", s->name);
		if(o < 0 || o >= nelem(h))
			mangle(pn);
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

	p = emallocz(sizeof(*p));
	p->as = o;
	p->lineno = BGETLE4(f);
	p->back = 2;
	zaddr(ctxt, pn, f, &p->from, h, &fromgotype);
	zaddr(ctxt, pn, f, &p->to, h, nil);

	if(ctxt->debugread)
		print("%P\n", p);

	switch(p->as) {
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
		if(s->type == 0 || s->type == SXREF) {
			s->type = SBSS;
			s->size = 0;
		}
		if(s->type != SBSS && s->type != SNOPTRBSS && !s->dupok) {
			ctxt->diag("%s: redefinition: %s in %s",
				pn, s->name, ctxt->cursym ? ctxt->cursym->name : "<none>");
			s->type = SBSS;
			s->size = 0;
		}
		if(p->to.offset > s->size)
			s->size = p->to.offset;
		if(p->from.scale & DUPOK)
			s->dupok = 1;
		if(p->from.scale & RODATA)
			s->type = SRODATA;
		else if(p->from.scale & NOPTR)
			s->type = SNOPTRBSS;
		goto loop;

	case ADATA:
		// Assume that AGLOBL comes after ADATA.
		// If we've seen an AGLOBL that said this sym was DUPOK,
		// ignore any more ADATA we see, which must be
		// redefinitions.
		s = p->from.sym;
		if(s->dupok) {
//			if(ctxt->debugvlog)
//				Bprint(ctxt->bso, "skipping %s in %s: dupok\n", s->name, pn);
			goto loop;
		}
		if(s->file == nil)
			s->file = pn;
		else if(s->file != pn) {
			ctxt->diag("multiple initialization for %s: in both %s and %s", s->name, s->file, pn);
			sysfatal("multiple init");
		}
		savedata(ctxt, s, p, pn);
		free(p);
		goto loop;

	case AGOK:
		ctxt->diag("%s: GOK opcode in %s", pn, ctxt->cursym ? ctxt->cursym->name : "<none>");
		ctxt->pc++;
		goto loop;

	case ATYPE:
		if(skip)
			goto casdef;
		ctxt->pc++;
		goto loop;

	case ATEXT:
		s = p->from.sym;
		if(s->text != nil) {
			if(p->from.scale & DUPOK) {
				skip = 1;
				goto casdef;
			}
			ctxt->diag("%s: %s: redefinition", pn, s->name);
			return;
		}
		if(ntext++ == 0 && s->type != 0 && s->type != SXREF) {
			/* redefinition, so file has probably been seen before */
			if(ctxt->debugvlog)
				ctxt->diag("skipping: %s: redefinition: %s", pn, s->name);
			return;
		}
		if(ctxt->cursym != nil && ctxt->cursym->text)
			finish(ctxt);
		skip = 0;
		if(ctxt->etextp)
			ctxt->etextp->next = s;
		else
			ctxt->textp = s;
		ctxt->etextp = s;
		s->text = p;
		ctxt->cursym = s;
		if(s->type != 0 && s->type != SXREF) {
			if(p->from.scale & DUPOK) {
				skip = 1;
				goto casdef;
			}
			ctxt->diag("%s: redefinition: %s\n%P", pn, s->name, p);
		}
		s->type = STEXT;
		s->hist = gethist(ctxt);
		s->value = ctxt->pc;
		s->args = p->to.offset2;
		lastp = p;
		p->pc = ctxt->pc++;
		goto loop;

	case AFMOVF:
	case AFADDF:
	case AFSUBF:
	case AFSUBRF:
	case AFMULF:
	case AFDIVF:
	case AFDIVRF:
	case AFCOMF:
	case AFCOMFP:
	case AMOVSS:
	case AADDSS:
	case ASUBSS:
	case AMULSS:
	case ADIVSS:
	case ACOMISS:
	case AUCOMISS:
		if(skip)
			goto casdef;
		if(p->from.type == D_FCONST) {
			/* size sb 9 max */
			sprint(literal, "$(%.17gf)", (float32)p->from.u.dval);
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
			p->from.type = D_EXTERN;
			p->from.sym = s;
			p->from.offset = 0;
		}
		goto casdef;

	case AFMOVD:
	case AFADDD:
	case AFSUBD:
	case AFSUBRD:
	case AFMULD:
	case AFDIVD:
	case AFDIVRD:
	case AFCOMD:
	case AFCOMDP:
	case AMOVSD:
	case AADDSD:
	case ASUBSD:
	case AMULSD:
	case ADIVSD:
	case ACOMISD:
	case AUCOMISD:
		if(skip)
			goto casdef;
		if(p->from.type == D_FCONST) {
			/* size sb 18 max */
			sprint(literal, "$%.17g",
				p->from.u.dval);
			s = linklookup(ctxt, literal, 0);
			if(s->type == 0) {
				int64 i64;
				s->type = SRODATA;
				memmove(&i64, &p->from.u.dval, 8);
				adduint64(ctxt, s, i64);
				s->reachable = 0;
			}
			p->from.type = D_EXTERN;
			p->from.sym = s;
			p->from.offset = 0;
		}
		goto casdef;

	casdef:
	default:
		if(skip)
			nopout8(p);
		p->pc = ctxt->pc;
		ctxt->pc++;

		if(p->to.type == D_BRANCH)
			p->to.offset += ipc;
		if(lastp == nil) {
			if(p->as != ANOP)
				ctxt->diag("unexpected instruction: %P", p);
			goto loop;
		}
		lastp->link = p;
		lastp = p;
		goto loop;
	}

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
