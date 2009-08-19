// Derived from Inferno utils/5c/swt.c
// http://code.google.com/p/inferno-os/source/browse/utils/5c/swt.c
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

#include "gg.h"

void
zname(Biobuf *b, Sym *s, int t)
{
	char *n;

	Bputc(b, ANAME);	/* as */
	Bputc(b, t);		/* type */
	Bputc(b, s->sym);	/* sym */

	for(n=s->package; *n; n++)
		Bputc(b, *n);
	Bputdot(b);
	for(n=s->name; *n; n++)
		Bputc(b, *n);
	Bputc(b, 0);
}

void
zfile(Biobuf *b, char *p, int n)
{
	Bputc(b, ANAME);
	Bputc(b, D_FILE);
	Bputc(b, 1);
	Bputc(b, '<');
	Bwrite(b, p, n);
	Bputc(b, 0);
}

void
zhist(Biobuf *b, int line, vlong offset)
{
	Addr a;

	Bputc(b, AHISTORY);
	Bputc(b, C_SCOND_NONE);
	Bputc(b, NREG);
	Bputc(b, line);
	Bputc(b, line>>8);
	Bputc(b, line>>16);
	Bputc(b, line>>24);
	zaddr(b, &zprog.from, 0);
	a = zprog.to;
	if(offset != 0) {
		a.offset = offset;
		a.type = D_CONST;
	}
	zaddr(b, &a, 0);
}

void
zaddr(Biobuf *b, Addr *a, int s)
{
	int32 l;
//	Ieee e;
	int i;
	char *n;

	switch(a->type) {
	case D_STATIC:
	case D_AUTO:
	case D_EXTERN:
	case D_PARAM:
		// TODO(kaib): remove once everything seems to work
		fatal("We should no longer generate these as types");

	default:
		Bputc(b, a->type);
		Bputc(b, a->reg);
		Bputc(b, s);
		Bputc(b, a->name);
	}

	switch(a->type) {
	default:
		print("unknown type %d in zaddr\n", a->type);

	case D_NONE:
	case D_REG:
	case D_FREG:
	case D_PSR:
		break;

	case D_CONST2:
		l = a->offset2;
		Bputc(b, l);
		Bputc(b, l>>8);
		Bputc(b, l>>16);
		Bputc(b, l>>24); // fall through
	case D_OREG:
	case D_CONST:
	case D_BRANCH:
	case D_SHIFT:
	case D_STATIC:
	case D_AUTO:
	case D_EXTERN:
	case D_PARAM:
		l = a->offset;
		Bputc(b, l);
		Bputc(b, l>>8);
		Bputc(b, l>>16);
		Bputc(b, l>>24);
		break;

	case D_SCONST:
		n = a->sval;
		for(i=0; i<NSNAME; i++) {
			Bputc(b, *n);
			n++;
		}
		break;

	case D_FCONST:
		fatal("zaddr D_FCONST not implemented");
		//ieeedtod(&e, a->dval);
		//l = e.l;
		//Bputc(b, l);
		//Bputc(b, l>>8);
		//Bputc(b, l>>16);
		//Bputc(b, l>>24);
		//l = e.h;
		//Bputc(b, l);
		//Bputc(b, l>>8);
		//Bputc(b, l>>16);
		//Bputc(b, l>>24);
		//break;
	}
}

void
dumpfuncs(void)
{
	Plist *pl;
	int sf, st, t, sym;
	struct { Sym *sym; short type; } h[NSYM];
	Sym *s;
	Prog *p;

	for(sym=0; sym<NSYM; sym++) {
		h[sym].sym = S;
		h[sym].type = 0;
	}
	sym = 1;

	// fix up pc
	pcloc = 0;
	for(pl=plist; pl!=nil; pl=pl->link) {
		for(p=pl->firstpc; p!=P; p=p->link) {
			p->loc = pcloc;
			if(p->as != ADATA && p->as != AGLOBL)
				pcloc++;
		}
	}

	// put out functions
	for(pl=plist; pl!=nil; pl=pl->link) {

		if(debug['S']) {
			s = S;
			if(pl->name != N)
				s = pl->name->sym;
			print("\n--- prog list \"%S\" ---\n", s);
			for(p=pl->firstpc; p!=P; p=p->link)
				print("%P\n", p);
		}

		for(p=pl->firstpc; p!=P; p=p->link) {
		jackpot:
			sf = 0;
			s = p->from.sym;
			while(s != S) {
				sf = s->sym;
				if(sf < 0 || sf >= NSYM)
					sf = 0;
				t = p->from.type;
				if(t == D_ADDR)
					t = p->from.name;
				if(h[sf].type == t)
				if(h[sf].sym == s)
					break;
				s->sym = sym;
				zname(bout, s, t);
				h[sym].sym = s;
				h[sym].type = t;
				sf = sym;
				sym++;
				if(sym >= NSYM)
					sym = 1;
				break;
			}
			st = 0;
			s = p->to.sym;
			while(s != S) {
				st = s->sym;
				if(st < 0 || st >= NSYM)
					st = 0;
				t = p->to.type;
				if(t == D_ADDR)
					t = p->to.name;
				if(h[st].type == t)
				if(h[st].sym == s)
					break;
				s->sym = sym;
				zname(bout, s, t);
				h[sym].sym = s;
				h[sym].type = t;
				st = sym;
				sym++;
				if(sym >= NSYM)
					sym = 1;
				if(st == sf)
					goto jackpot;
				break;
			}
			Bputc(bout, p->as);
			Bputc(bout, p->scond);
 			Bputc(bout, p->reg);
			Bputc(bout, p->lineno);
			Bputc(bout, p->lineno>>8);
			Bputc(bout, p->lineno>>16);
			Bputc(bout, p->lineno>>24);
			zaddr(bout, &p->from, sf);
			zaddr(bout, &p->to, st);
		}
	}
}

/* deferred DATA output */
static Prog *strdat;
static Prog *estrdat;
static int gflag;
static Prog *savepc;

static void
data(void)
{
	gflag = debug['g'];
	debug['g'] = 0;

	if(estrdat == nil) {
		strdat = mal(sizeof(*pc));
		clearp(strdat);
		estrdat = strdat;
	}
	if(savepc)
		fatal("data phase error");
	savepc = pc;
	pc = estrdat;
}

static void
text(void)
{
	if(!savepc)
		fatal("text phase error");
	debug['g'] = gflag;
	estrdat = pc;
	pc = savepc;
	savepc = nil;
}

void
dumpdata(void)
{
	Prog *p;

	if(estrdat == nil)
		return;
	*pc = *strdat;
	if(gflag)
		for(p=pc; p!=estrdat; p=p->link)
			print("%P\n", p);
	pc = estrdat;
}

/*
 * make a refer to the data s, s+len
 * emitting DATA if needed.
 */
void
datastring(char *s, int len, Addr *a)
{
	int w;
	Prog *p;
	Addr ac, ao;
	static int gen;
	struct {
		Strlit lit;
		char buf[100];
	} tmp;

	// string
	memset(&ao, 0, sizeof(ao));
	ao.type = D_OREG;
	ao.name = D_STATIC;
	ao.etype = TINT32;
	ao.offset = 0;		// fill in
	ao.reg = NREG;

	// constant
	memset(&ac, 0, sizeof(ac));
	ac.type = D_CONST;
	ac.name = D_NONE;
	ac.offset = 0;		// fill in
	ac.reg = NREG;

	// huge strings are made static to avoid long names.
	if(len > 100) {
		snprint(namebuf, sizeof(namebuf), ".string.%d", gen++);
		ao.sym = lookup(namebuf);
		ao.name = D_STATIC;
	} else {
		if(len > 0 && s[len-1] == '\0')
			len--;
		tmp.lit.len = len;
		memmove(tmp.lit.s, s, len);
		tmp.lit.s[len] = '\0';
		len++;
		snprint(namebuf, sizeof(namebuf), "\"%Z\"", &tmp.lit);
		ao.sym = pkglookup(namebuf, "string");
		ao.name = D_EXTERN;
	}
	*a = ao;

	// only generate data the first time.
	if(ao.sym->flags & SymUniq)
		return;
	ao.sym->flags |= SymUniq;

	data();
	for(w=0; w<len; w+=8) {
		p = pc;
		gins(ADATA, N, N);

		// DATA s+w, [NSNAME], $"xxx"
		p->from = ao;
		p->from.offset = w;

		p->reg = NSNAME;
		if(w+8 > len)
			p->reg = len-w;

		p->to = ac;
		p->to.type = D_SCONST;
		p->to.offset = len;
		memmove(p->to.sval, s+w, p->reg);
	}
	p = pc;
	ggloblsym(ao.sym, len, ao.name == D_EXTERN);
	if(ao.name == D_STATIC)
		p->from.name = D_STATIC;
	text();
}

/*
 * make a refer to the string sval,
 * emitting DATA if needed.
 */
void
datagostring(Strlit *sval, Addr *a)
{
	Prog *p;
	Addr ac, ao, ap;
	int32 wi, wp;
	static int gen;

	memset(&ac, 0, sizeof(ac));
	memset(&ao, 0, sizeof(ao));
	memset(&ap, 0, sizeof(ap));

	// constant
	ac.type = D_CONST;
	ac.name = D_NONE;
	ac.offset = 0;			// fill in
	ac.reg = NREG;

	// string len+ptr
	ao.type = D_OREG;
	ao.name = D_STATIC;		// fill in
	ao.etype = TINT32;
	ao.sym = nil;			// fill in
	ao.reg = NREG;

	// $string len+ptr
	datastring(sval->s, sval->len, &ap);

	ap.type = D_CONST;
	ap.etype = TINT32;
	wi = types[TUINT32]->width;
	wp = types[tptr]->width;

	if(ap.name == D_STATIC) {
		// huge strings are made static to avoid long names
		snprint(namebuf, sizeof(namebuf), ".gostring.%d", ++gen);
		ao.sym = lookup(namebuf);
		ao.name = D_STATIC;
	} else {
		// small strings get named by their contents,
		// so that multiple modules using the same string
		// can share it.
		snprint(namebuf, sizeof(namebuf), "\"%Z\"", sval);
		ao.sym = pkglookup(namebuf, "go.string");
		ao.name = D_EXTERN;
	}

	*a = ao;
	if(ao.sym->flags & SymUniq)
		return;
	ao.sym->flags |= SymUniq;

	data();
	// DATA gostring, wp, $cstring
	p = pc;
	gins(ADATA, N, N);
	p->from = ao;
	p->reg = wp;
	p->to = ap;

	// DATA gostring+wp, wi, $len
	p = pc;
	gins(ADATA, N, N);
	p->from = ao;
	p->from.offset = wp;
	p->reg = wi;
	p->to = ac;
	p->to.offset = sval->len;

	p = pc;
	ggloblsym(ao.sym, types[TSTRING]->width, ao.type == D_EXTERN);
	if(ao.name == D_STATIC)
		p->from.name = D_STATIC;
	text();
}

int
dstringptr(Sym *s, int off, char *str)
{
	Prog *p;

	off = rnd(off, widthptr);
	p = gins(ADATA, N, N);
	p->from.type = D_OREG;
	p->from.name = D_EXTERN;
	p->from.sym = s;
	p->from.offset = off;
	p->reg = widthptr;

	datastring(str, strlen(str)+1, &p->to);
	p->to.type = D_CONST;
	p->to.etype = TINT32;
	off += widthptr;

	return off;
}

int
dgostrlitptr(Sym *s, int off, Strlit *lit)
{
	Prog *p;

	if(lit == nil)
		return duintptr(s, off, 0);

	off = rnd(off, widthptr);
	p = gins(ADATA, N, N);
	p->from.type = D_OREG;
	p->from.name = D_EXTERN;
	p->from.sym = s;
	p->from.offset = off;
	p->from.reg = widthptr;
	datagostring(lit, &p->to);
	p->to.type = D_CONST;
	p->to.etype = TINT32;
	off += widthptr;

	return off;
}

int
dgostringptr(Sym *s, int off, char *str)
{
	int n;
	Strlit *lit;

	if(str == nil)
		return duintptr(s, off, 0);

	n = strlen(str);
	lit = mal(sizeof *lit + n);
	strcpy(lit->s, str);
	lit->len = n;
	return dgostrlitptr(s, off, lit);
}

int
duintxx(Sym *s, int off, uint64 v, int wid)
{
	Prog *p;

	off = rnd(off, wid);

	p = gins(ADATA, N, N);
	p->from.type = D_OREG;
	p->from.name = D_EXTERN;
	p->from.sym = s;
	p->from.offset = off;
	p->reg = wid;
	p->to.type = D_CONST;
	p->to.name = D_NONE;
	p->to.offset = v;
	off += wid;

	return off;
}

int
dsymptr(Sym *s, int off, Sym *x, int xoff)
{
	Prog *p;

	off = rnd(off, widthptr);

	p = gins(ADATA, N, N);
	p->from.type = D_OREG;
	p->from.name = D_EXTERN;
	p->from.sym = s;
	p->from.offset = off;
	p->reg = widthptr;
	p->to.type = D_CONST;
	p->to.name = D_EXTERN;
	p->to.sym = x;
	p->to.offset = xoff;
	off += widthptr;

	return off;
}


void
genembedtramp(Type *rcvr, Type *method, Sym *newnam)
{
	Sym *e;
	int c, d, o;
	Prog *p;
	Type *f;

	e = method->sym;
	for(d=0; d<nelem(dotlist); d++) {
		c = adddot1(e, rcvr, d, nil);
		if(c == 1)
			goto out;
	}
	fatal("genembedtramp %T.%S", rcvr, method->sym);

out:
	newplist()->name = newname(newnam);

	//TEXT	main·S_test2(SB),7,$0
	p = pc;
	gins(ATEXT, N, N);
	p->from.type = D_OREG;
	p->from.name = D_EXTERN;
	p->from.sym = newnam;
	p->to.type = D_CONST2;
	p->reg = 7;
	p->to.offset2 = 0;
	p->to.reg = NREG;
print("1. %P\n", p);

	o = 0;
	for(c=d-1; c>=0; c--) {
		f = dotlist[c].field;
		o += f->width;
		if(!isptr[f->type->etype])
			continue;

		//MOVW	o(R0), R0
		p = pc;
		gins(AMOVW, N, N);
		p->from.type = D_OREG;
		p->from.reg = REGARG;
		p->from.offset = o;
		p->to.type = D_REG;
		p->to.reg = REGARG;
print("2. %P\n", p);
		o = 0;
	}
	if(o != 0) {
		//MOVW	$XX(R0), R0
		p = pc;
		gins(AMOVW, N, N);
		p->from.type = D_CONST;
		p->from.reg = REGARG;
		p->from.offset = o;
		p->to.type = D_REG;
		p->to.reg = REGARG;
print("3. %P\n", p);
	}

	f = dotlist[0].field;
	//B	main·*Sub_test2(SB)
	if(isptr[f->type->etype])
		f = f->type;
	p = pc;
	gins(AB, N, N);
	p->to.type = D_OREG;
	p->to.reg = NREG;
	p->to.name = D_EXTERN;
	p->to.sym = methodsym(method->sym, ptrto(f->type));
print("4. %P\n", p);

	pc->as = ARET;	// overwrite AEND
}

void
nopout(Prog *p)
{
	p->as = ANOP;
}
