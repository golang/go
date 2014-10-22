// Inferno utils/8c/txt.c
// http://code.google.com/p/inferno-os/source/browse/utils/8c/txt.c
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

#include "gc.h"


int thechar = '8';
char *thestring = "386";

LinkArch	*thelinkarch = &link386;

void
linkarchinit(void)
{
}

void
ginit(void)
{
	int i;
	Type *t;

	exregoffset = 0;
	exfregoffset = 0;
	listinit();
	nstring = 0;
	mnstring = 0;
	nrathole = 0;
	pc = 0;
	breakpc = -1;
	continpc = -1;
	cases = C;
	lastp = P;
	tfield = types[TLONG];

	zprog.link = P;
	zprog.as = AGOK;
	zprog.from.type = D_NONE;
	zprog.from.index = D_NONE;
	zprog.from.scale = 0;
	zprog.to = zprog.from;

	regnode.op = OREGISTER;
	regnode.class = CEXREG;
	regnode.reg = REGTMP;
	regnode.complex = 0;
	regnode.addable = 11;
	regnode.type = types[TLONG];

	fregnode0 = regnode;
	fregnode0.reg = D_F0;
	fregnode0.type = types[TDOUBLE];

	fregnode1 = fregnode0;
	fregnode1.reg = D_F0+1;

	constnode.op = OCONST;
	constnode.class = CXXX;
	constnode.complex = 0;
	constnode.addable = 20;
	constnode.type = types[TLONG];

	fconstnode.op = OCONST;
	fconstnode.class = CXXX;
	fconstnode.complex = 0;
	fconstnode.addable = 20;
	fconstnode.type = types[TDOUBLE];

	nodsafe = new(ONAME, Z, Z);
	nodsafe->sym = slookup(".safe");
	nodsafe->type = types[TINT];
	nodsafe->etype = types[TINT]->etype;
	nodsafe->class = CAUTO;
	complex(nodsafe);

	t = typ(TARRAY, types[TCHAR]);
	symrathole = slookup(".rathole");
	symrathole->class = CGLOBL;
	symrathole->type = t;

	nodrat = new(ONAME, Z, Z);
	nodrat->sym = symrathole;
	nodrat->type = types[TIND];
	nodrat->etype = TVOID;
	nodrat->class = CGLOBL;
	complex(nodrat);
	nodrat->type = t;

	nodret = new(ONAME, Z, Z);
	nodret->sym = slookup(".ret");
	nodret->type = types[TIND];
	nodret->etype = TIND;
	nodret->class = CPARAM;
	nodret = new(OIND, nodret, Z);
	complex(nodret);

	com64init();

	for(i=0; i<nelem(reg); i++) {
		reg[i] = 1;
		if(i >= D_AX && i <= D_DI && i != D_SP)
			reg[i] = 0;
	}
}

void
gclean(void)
{
	int i;
	Sym *s;

	reg[D_SP]--;
	for(i=D_AX; i<=D_DI; i++)
		if(reg[i])
			diag(Z, "reg %R left allocated", i);
	while(mnstring)
		outstring("", 1L);
	symstring->type->width = nstring;
	symrathole->type->width = nrathole;
	for(i=0; i<NHASH; i++)
	for(s = hash[i]; s != S; s = s->link) {
		if(s->type == T)
			continue;
		if(s->type->width == 0)
			continue;
		if(s->class != CGLOBL && s->class != CSTATIC)
			continue;
		if(s->type == types[TENUM])
			continue;
		gpseudo(AGLOBL, s, nodconst(s->type->width));
	}
	nextpc();
	p->as = AEND;
	outcode();
}

void
nextpc(void)
{
	Plist *pl;

	p = alloc(sizeof(*p));
	*p = zprog;
	p->lineno = nearln;
	p->pc = pc;
	pc++;
	if(lastp == nil) {
		pl = linknewplist(ctxt);
		pl->firstpc = p;
	} else
		lastp->link = p;
	lastp = p;
}

void
gargs(Node *n, Node *tn1, Node *tn2)
{
	int32 regs;
	Node fnxargs[20], *fnxp;

	regs = cursafe;

	fnxp = fnxargs;
	garg1(n, tn1, tn2, 0, &fnxp);	/* compile fns to temps */

	curarg = 0;
	fnxp = fnxargs;
	garg1(n, tn1, tn2, 1, &fnxp);	/* compile normal args and temps */

	cursafe = regs;
}

int
nareg(void)
{
	int i, n;

	n = 0;
	for(i=D_AX; i<=D_DI; i++)
		if(reg[i] == 0)
			n++;
	return n;
}

void
garg1(Node *n, Node *tn1, Node *tn2, int f, Node **fnxp)
{
	Node nod;

	if(n == Z)
		return;
	if(n->op == OLIST) {
		garg1(n->left, tn1, tn2, f, fnxp);
		garg1(n->right, tn1, tn2, f, fnxp);
		return;
	}
	if(f == 0) {
		if(n->complex >= FNX) {
			regsalloc(*fnxp, n);
			nod = znode;
			nod.op = OAS;
			nod.left = *fnxp;
			nod.right = n;
			nod.type = n->type;
			cgen(&nod, Z);
			(*fnxp)++;
		}
		return;
	}
	if(typesu[n->type->etype] || typev[n->type->etype]) {
		regaalloc(tn2, n);
		if(n->complex >= FNX) {
			sugen(*fnxp, tn2, n->type->width);
			(*fnxp)++;
		} else
			sugen(n, tn2, n->type->width);
		return;
	}
	if(REGARG >= 0 && curarg == 0 && typeilp[n->type->etype]) {
		regaalloc1(tn1, n);
		if(n->complex >= FNX) {
			cgen(*fnxp, tn1);
			(*fnxp)++;
		} else
			cgen(n, tn1);
		return;
	}
	if(vconst(n) == 0) {
		regaalloc(tn2, n);
		gmove(n, tn2);
		return;
	}
	regalloc(tn1, n, Z);
	if(n->complex >= FNX) {
		cgen(*fnxp, tn1);
		(*fnxp)++;
	} else
		cgen(n, tn1);
	regaalloc(tn2, n);
	gmove(tn1, tn2);
	regfree(tn1);
}

Node*
nodconst(int32 v)
{
	constnode.vconst = v;
	return &constnode;
}

Node*
nodfconst(double d)
{
	fconstnode.fconst = d;
	return &fconstnode;
}

int
isreg(Node *n, int r)
{

	if(n->op == OREGISTER)
		if(n->reg == r)
			return 1;
	return 0;
}

int
nodreg(Node *n, Node *nn, int r)
{

	*n = regnode;
	n->reg = r;
	if(reg[r] == 0)
		return 0;
	if(nn != Z) {
		n->type = nn->type;
		n->lineno = nn->lineno;
		if(nn->op == OREGISTER)
		if(nn->reg == r)
			return 0;
	}
	return 1;
}

void
regret(Node *n, Node *nn, Type *t, int mode)
{
	int r;

	if(mode == 0 || hasdotdotdot(t) || nn->type->width == 0) {
		r = REGRET;
		if(typefd[nn->type->etype])
			r = FREGRET;
		nodreg(n, nn, r);
		reg[r]++;
		return;
	}
	
	if(mode == 1) {
		// fetch returned value after call.
		// already called gargs, so curarg is set.
		curarg = (curarg+3) & ~3;
		regaalloc(n, nn);
		return;
	}
	
	if(mode == 2) {
		// store value to be returned.
		// must compute arg offset.
		if(t->etype != TFUNC)
			fatal(Z, "bad regret func %T", t);
		*n = *nn;
		n->op = ONAME;
		n->class = CPARAM;
		n->sym = slookup(".retx");
		n->complex = 0;
		n->addable = 20;
		n->xoffset = argsize(0);
		return;
	}
	
	fatal(Z, "bad regret");
}

void
regalloc(Node *n, Node *tn, Node *o)
{
	int i;

	switch(tn->type->etype) {
	case TCHAR:
	case TUCHAR:
	case TSHORT:
	case TUSHORT:
	case TINT:
	case TUINT:
	case TLONG:
	case TULONG:
	case TIND:
		if(o != Z && o->op == OREGISTER) {
			i = o->reg;
			if(i >= D_AX && i <= D_DI)
				goto out;
		}
		for(i=D_AX; i<=D_DI; i++)
			if(reg[i] == 0)
				goto out;
		diag(tn, "out of fixed registers");
		goto err;

	case TFLOAT:
	case TDOUBLE:
	case TVLONG:
		i = D_F0;
		goto out;
	}
	diag(tn, "unknown type in regalloc: %T", tn->type);
err:
	i = 0;
out:
	if(i)
		reg[i]++;
	nodreg(n, tn, i);
}

void
regialloc(Node *n, Node *tn, Node *o)
{
	Node nod;

	nod = *tn;
	nod.type = types[TIND];
	regalloc(n, &nod, o);
}

void
regfree(Node *n)
{
	int i;

	i = 0;
	if(n->op != OREGISTER && n->op != OINDREG)
		goto err;
	i = n->reg;
	if(i < 0 || i >= nelem(reg))
		goto err;
	if(reg[i] <= 0)
		goto err;
	reg[i]--;
	return;
err:
	diag(n, "error in regfree: %R", i);
}

void
regsalloc(Node *n, Node *nn)
{
	cursafe = align(cursafe, nn->type, Aaut3, nil);
	maxargsafe = maxround(maxargsafe, cursafe+curarg);
	*n = *nodsafe;
	n->xoffset = -(stkoff + cursafe);
	n->type = nn->type;
	n->etype = nn->type->etype;
	n->lineno = nn->lineno;
}

void
regaalloc1(Node *n, Node *nn)
{
	if(REGARG < 0) {
		fatal(n, "regaalloc1 and REGARG<0");
		return;
	}
	nodreg(n, nn, REGARG);
	reg[REGARG]++;
	curarg = align(curarg, nn->type, Aarg1, nil);
	curarg = align(curarg, nn->type, Aarg2, nil);
	maxargsafe = maxround(maxargsafe, cursafe+curarg);
}

void
regaalloc(Node *n, Node *nn)
{
	curarg = align(curarg, nn->type, Aarg1, nil);
	*n = *nn;
	n->op = OINDREG;
	n->reg = REGSP;
	n->xoffset = curarg;
	n->complex = 0;
	n->addable = 20;
	curarg = align(curarg, nn->type, Aarg2, nil);
	maxargsafe = maxround(maxargsafe, cursafe+curarg);
}

void
regind(Node *n, Node *nn)
{

	if(n->op != OREGISTER) {
		diag(n, "regind not OREGISTER");
		return;
	}
	n->op = OINDREG;
	n->type = nn->type;
}

void
naddr(Node *n, Addr *a)
{
	int32 v;

	a->type = D_NONE;
	if(n == Z)
		return;
	switch(n->op) {
	default:
	bad:
		diag(n, "bad in naddr: %O %D", n->op, a);
		break;

	case OREGISTER:
		a->type = n->reg;
		a->sym = nil;
		break;

	case OEXREG:
		a->type = D_INDIR + D_TLS;
		a->offset = n->reg - 1;
		break;

	case OIND:
		naddr(n->left, a);
		if(a->type >= D_AX && a->type <= D_DI)
			a->type += D_INDIR;
		else
		if(a->type == D_CONST)
			a->type = D_NONE+D_INDIR;
		else
		if(a->type == D_ADDR) {
			a->type = a->index;
			a->index = D_NONE;
		} else
			goto bad;
		break;

	case OINDEX:
		a->type = idx.ptr;
		if(n->left->op == OADDR || n->left->op == OCONST)
			naddr(n->left, a);
		if(a->type >= D_AX && a->type <= D_DI)
			a->type += D_INDIR;
		else
		if(a->type == D_CONST)
			a->type = D_NONE+D_INDIR;
		else
		if(a->type == D_ADDR) {
			a->type = a->index;
			a->index = D_NONE;
		} else
			goto bad;
		a->index = idx.reg;
		a->scale = n->scale;
		a->offset += n->xoffset;
		break;

	case OINDREG:
		a->type = n->reg+D_INDIR;
		a->sym = nil;
		a->offset = n->xoffset;
		break;

	case ONAME:
		a->etype = n->etype;
		a->type = D_STATIC;
		a->sym = linksym(n->sym);
		a->offset = n->xoffset;
		if(n->class == CSTATIC)
			break;
		if(n->class == CEXTERN || n->class == CGLOBL) {
			a->type = D_EXTERN;
			break;
		}
		if(n->class == CAUTO) {
			a->type = D_AUTO;
			break;
		}
		if(n->class == CPARAM) {
			a->type = D_PARAM;
			break;
		}
		goto bad;

	case OCONST:
		if(typefd[n->type->etype]) {
			a->type = D_FCONST;
			a->u.dval = n->fconst;
			break;
		}
		a->sym = nil;
		a->type = D_CONST;
		a->offset = n->vconst;
		break;

	case OADDR:
		naddr(n->left, a);
		if(a->type >= D_INDIR) {
			a->type -= D_INDIR;
			break;
		}
		if(a->type == D_EXTERN || a->type == D_STATIC ||
		   a->type == D_AUTO || a->type == D_PARAM)
			if(a->index == D_NONE) {
				a->index = a->type;
				a->type = D_ADDR;
				break;
			}
		goto bad;

	case OADD:
		if(n->right->op == OCONST) {
			v = n->right->vconst;
			naddr(n->left, a);
		} else
		if(n->left->op == OCONST) {
			v = n->left->vconst;
			naddr(n->right, a);
		} else
			goto bad;
		a->offset += v;
		break;

	}
}

#define	CASE(a,b)	((a<<8)|(b<<0))

void
gmove(Node *f, Node *t)
{
	int ft, tt, a;
	Node nod, nod1;
	Prog *p1;

	ft = f->type->etype;
	tt = t->type->etype;
	if(debug['M'])
		print("gop: %O %O[%s],%O[%s]\n", OAS,
			f->op, tnames[ft], t->op, tnames[tt]);
	if(typefd[ft] && f->op == OCONST) {
		if(f->fconst == 0)
			gins(AFLDZ, Z, Z);
		else
		if(f->fconst == 1)
			gins(AFLD1, Z, Z);
		else
			gins(AFMOVD, f, &fregnode0);
		gmove(&fregnode0, t);
		return;
	}
/*
 * load
 */
	if(f->op == ONAME || f->op == OINDREG ||
	   f->op == OIND || f->op == OINDEX)
	switch(ft) {
	case TCHAR:
		a = AMOVBLSX;
		goto ld;
	case TUCHAR:
		a = AMOVBLZX;
		goto ld;
	case TSHORT:
		if(typefd[tt]) {
			gins(AFMOVW, f, &fregnode0);
			gmove(&fregnode0, t);
			return;
		}
		a = AMOVWLSX;
		goto ld;
	case TUSHORT:
		a = AMOVWLZX;
		goto ld;
	case TINT:
	case TUINT:
	case TLONG:
	case TULONG:
	case TIND:
		if(typefd[tt]) {
			gins(AFMOVL, f, &fregnode0);
			gmove(&fregnode0, t);
			return;
		}
		a = AMOVL;

	ld:
		regalloc(&nod, f, t);
		nod.type = types[TLONG];
		gins(a, f, &nod);
		gmove(&nod, t);
		regfree(&nod);
		return;

	case TFLOAT:
		gins(AFMOVF, f, t);
		return;
	case TDOUBLE:
		gins(AFMOVD, f, t);
		return;
	case TVLONG:
		gins(AFMOVV, f, t);
		return;
	}

/*
 * store
 */
	if(t->op == ONAME || t->op == OINDREG ||
	   t->op == OIND || t->op == OINDEX)
	switch(tt) {
	case TCHAR:
	case TUCHAR:
		a = AMOVB;	goto st;
	case TSHORT:
	case TUSHORT:
		a = AMOVW;	goto st;
	case TINT:
	case TUINT:
	case TLONG:
	case TULONG:
	case TIND:
		a = AMOVL;	goto st;

	st:
		if(f->op == OCONST) {
			gins(a, f, t);
			return;
		}
		regalloc(&nod, t, f);
		gmove(f, &nod);
		gins(a, &nod, t);
		regfree(&nod);
		return;

	case TFLOAT:
		gins(AFMOVFP, f, t);
		return;
	case TDOUBLE:
		gins(AFMOVDP, f, t);
		return;
	case TVLONG:
		gins(AFMOVVP, f, t);
		return;
	}

/*
 * convert
 */
	switch(CASE(ft,tt)) {
	default:
/*
 * integer to integer
 ********
		a = AGOK;	break;

	case CASE(	TCHAR,	TCHAR):
	case CASE(	TUCHAR,	TCHAR):
	case CASE(	TSHORT,	TCHAR):
	case CASE(	TUSHORT,TCHAR):
	case CASE(	TINT,	TCHAR):
	case CASE(	TUINT,	TCHAR):
	case CASE(	TLONG,	TCHAR):
	case CASE(	TULONG,	TCHAR):
	case CASE(	TIND,	TCHAR):

	case CASE(	TCHAR,	TUCHAR):
	case CASE(	TUCHAR,	TUCHAR):
	case CASE(	TSHORT,	TUCHAR):
	case CASE(	TUSHORT,TUCHAR):
	case CASE(	TINT,	TUCHAR):
	case CASE(	TUINT,	TUCHAR):
	case CASE(	TLONG,	TUCHAR):
	case CASE(	TULONG,	TUCHAR):
	case CASE(	TIND,	TUCHAR):

	case CASE(	TSHORT,	TSHORT):
	case CASE(	TUSHORT,TSHORT):
	case CASE(	TINT,	TSHORT):
	case CASE(	TUINT,	TSHORT):
	case CASE(	TLONG,	TSHORT):
	case CASE(	TULONG,	TSHORT):
	case CASE(	TIND,	TSHORT):

	case CASE(	TSHORT,	TUSHORT):
	case CASE(	TUSHORT,TUSHORT):
	case CASE(	TINT,	TUSHORT):
	case CASE(	TUINT,	TUSHORT):
	case CASE(	TLONG,	TUSHORT):
	case CASE(	TULONG,	TUSHORT):
	case CASE(	TIND,	TUSHORT):

	case CASE(	TINT,	TINT):
	case CASE(	TUINT,	TINT):
	case CASE(	TLONG,	TINT):
	case CASE(	TULONG,	TINT):
	case CASE(	TIND,	TINT):

	case CASE(	TINT,	TUINT):
	case CASE(	TUINT,	TUINT):
	case CASE(	TLONG,	TUINT):
	case CASE(	TULONG,	TUINT):
	case CASE(	TIND,	TUINT):

	case CASE(	TINT,	TLONG):
	case CASE(	TUINT,	TLONG):
	case CASE(	TLONG,	TLONG):
	case CASE(	TULONG,	TLONG):
	case CASE(	TIND,	TLONG):

	case CASE(	TINT,	TULONG):
	case CASE(	TUINT,	TULONG):
	case CASE(	TLONG,	TULONG):
	case CASE(	TULONG,	TULONG):
	case CASE(	TIND,	TULONG):

	case CASE(	TINT,	TIND):
	case CASE(	TUINT,	TIND):
	case CASE(	TLONG,	TIND):
	case CASE(	TULONG,	TIND):
	case CASE(	TIND,	TIND):
 *****/
		a = AMOVL;
		break;

	case CASE(	TSHORT,	TINT):
	case CASE(	TSHORT,	TUINT):
	case CASE(	TSHORT,	TLONG):
	case CASE(	TSHORT,	TULONG):
	case CASE(	TSHORT,	TIND):
		a = AMOVWLSX;
		if(f->op == OCONST) {
			f->vconst &= 0xffff;
			if(f->vconst & 0x8000)
				f->vconst |= 0xffff0000;
			a = AMOVL;
		}
		break;

	case CASE(	TUSHORT,TINT):
	case CASE(	TUSHORT,TUINT):
	case CASE(	TUSHORT,TLONG):
	case CASE(	TUSHORT,TULONG):
	case CASE(	TUSHORT,TIND):
		a = AMOVWLZX;
		if(f->op == OCONST) {
			f->vconst &= 0xffff;
			a = AMOVL;
		}
		break;

	case CASE(	TCHAR,	TSHORT):
	case CASE(	TCHAR,	TUSHORT):
	case CASE(	TCHAR,	TINT):
	case CASE(	TCHAR,	TUINT):
	case CASE(	TCHAR,	TLONG):
	case CASE(	TCHAR,	TULONG):
	case CASE(	TCHAR,	TIND):
		a = AMOVBLSX;
		if(f->op == OCONST) {
			f->vconst &= 0xff;
			if(f->vconst & 0x80)
				f->vconst |= 0xffffff00;
			a = AMOVL;
		}
		break;

	case CASE(	TUCHAR,	TSHORT):
	case CASE(	TUCHAR,	TUSHORT):
	case CASE(	TUCHAR,	TINT):
	case CASE(	TUCHAR,	TUINT):
	case CASE(	TUCHAR,	TLONG):
	case CASE(	TUCHAR,	TULONG):
	case CASE(	TUCHAR,	TIND):
		a = AMOVBLZX;
		if(f->op == OCONST) {
			f->vconst &= 0xff;
			a = AMOVL;
		}
		break;

/*
 * float to fix
 */
	case CASE(	TFLOAT,	TCHAR):
	case CASE(	TFLOAT,	TUCHAR):
	case CASE(	TFLOAT,	TSHORT):
	case CASE(	TFLOAT,	TUSHORT):
	case CASE(	TFLOAT,	TINT):
	case CASE(	TFLOAT,	TUINT):
	case CASE(	TFLOAT,	TLONG):
	case CASE(	TFLOAT,	TULONG):
	case CASE(	TFLOAT,	TIND):

	case CASE(	TDOUBLE,TCHAR):
	case CASE(	TDOUBLE,TUCHAR):
	case CASE(	TDOUBLE,TSHORT):
	case CASE(	TDOUBLE,TUSHORT):
	case CASE(	TDOUBLE,TINT):
	case CASE(	TDOUBLE,TUINT):
	case CASE(	TDOUBLE,TLONG):
	case CASE(	TDOUBLE,TULONG):
	case CASE(	TDOUBLE,TIND):

	case CASE(	TVLONG,	TCHAR):
	case CASE(	TVLONG,	TUCHAR):
	case CASE(	TVLONG,	TSHORT):
	case CASE(	TVLONG,	TUSHORT):
	case CASE(	TVLONG,	TINT):
	case CASE(	TVLONG,	TUINT):
	case CASE(	TVLONG,	TLONG):
	case CASE(	TVLONG,	TULONG):
	case CASE(	TVLONG,	TIND):
		if(fproundflg) {
			regsalloc(&nod, &regnode);
			gins(AFMOVLP, f, &nod);
			gmove(&nod, t);
			return;
		}
		regsalloc(&nod, &regnode);
		regsalloc(&nod1, &regnode);
		gins(AFSTCW, Z, &nod1);
		nod1.xoffset += 2;
		gins(AMOVW, nodconst(0xf7f), &nod1);
		gins(AFLDCW, &nod1, Z);
		gins(AFMOVLP, f, &nod);
		nod1.xoffset -= 2;
		gins(AFLDCW, &nod1, Z);
		gmove(&nod, t);
		return;

/*
 * ulong to float
 */
	case CASE(	TULONG,	TDOUBLE):
	case CASE(	TULONG,	TVLONG):
	case CASE(	TULONG,	TFLOAT):
	case CASE(	TUINT,	TDOUBLE):
	case CASE(	TUINT,	TVLONG):
	case CASE(	TUINT,	TFLOAT):
		regalloc(&nod, f, f);
		gmove(f, &nod);
		regsalloc(&nod1, &regnode);
		gmove(&nod, &nod1);
		gins(AFMOVL, &nod1, &fregnode0);
		gins(ACMPL, &nod, nodconst(0));
		gins(AJGE, Z, Z);
		p1 = p;
		gins(AFADDD, nodfconst(4294967296.), &fregnode0);
		patch(p1, pc);
		regfree(&nod);
		return;

/*
 * fix to float
 */
	case CASE(	TCHAR,	TFLOAT):
	case CASE(	TUCHAR,	TFLOAT):
	case CASE(	TSHORT,	TFLOAT):
	case CASE(	TUSHORT,TFLOAT):
	case CASE(	TINT,	TFLOAT):
	case CASE(	TLONG,	TFLOAT):
	case CASE(	TIND,	TFLOAT):

	case CASE(	TCHAR,	TDOUBLE):
	case CASE(	TUCHAR,	TDOUBLE):
	case CASE(	TSHORT,	TDOUBLE):
	case CASE(	TUSHORT,TDOUBLE):
	case CASE(	TINT,	TDOUBLE):
	case CASE(	TLONG,	TDOUBLE):
	case CASE(	TIND,	TDOUBLE):

	case CASE(	TCHAR,	TVLONG):
	case CASE(	TUCHAR,	TVLONG):
	case CASE(	TSHORT,	TVLONG):
	case CASE(	TUSHORT,TVLONG):
	case CASE(	TINT,	TVLONG):
	case CASE(	TLONG,	TVLONG):
	case CASE(	TIND,	TVLONG):
		regsalloc(&nod, &regnode);
		gmove(f, &nod);
		gins(AFMOVL, &nod, &fregnode0);
		return;

/*
 * float to float
 */
	case CASE(	TFLOAT,	TFLOAT):
	case CASE(	TDOUBLE,TFLOAT):
	case CASE(	TVLONG,	TFLOAT):

	case CASE(	TFLOAT,	TDOUBLE):
	case CASE(	TDOUBLE,TDOUBLE):
	case CASE(	TVLONG,	TDOUBLE):

	case CASE(	TFLOAT,	TVLONG):
	case CASE(	TDOUBLE,TVLONG):
	case CASE(	TVLONG,	TVLONG):
		a = AFMOVD;	break;
	}
	if(a == AMOVL || a == AFMOVD)
	if(samaddr(f, t))
		return;
	gins(a, f, t);
}

void
doindex(Node *n)
{
	Node nod, nod1;
	int32 v;

if(debug['Y'])
prtree(n, "index");

if(n->left->complex >= FNX)
print("botch in doindex\n");

	regalloc(&nod, &regnode, Z);
	v = constnode.vconst;
	cgen(n->right, &nod);
	idx.ptr = D_NONE;
	if(n->left->op == OCONST)
		idx.ptr = D_CONST;
	else if(n->left->op == OREGISTER)
		idx.ptr = n->left->reg;
	else if(n->left->op != OADDR) {
		reg[D_BP]++;	// can't be used as a base
		regalloc(&nod1, &regnode, Z);
		cgen(n->left, &nod1);
		idx.ptr = nod1.reg;
		regfree(&nod1);
		reg[D_BP]--;
	}
	idx.reg = nod.reg;
	regfree(&nod);
	constnode.vconst = v;
}

void
gins(int a, Node *f, Node *t)
{

	if(f != Z && f->op == OINDEX)
		doindex(f);
	if(t != Z && t->op == OINDEX)
		doindex(t);
	nextpc();
	p->as = a;
	if(f != Z)
		naddr(f, &p->from);
	if(t != Z)
		naddr(t, &p->to);
	if(debug['g'])
		print("%P\n", p);
}

void
fgopcode(int o, Node *f, Node *t, int pop, int rev)
{
	int a, et;
	Node nod;

	et = TLONG;
	if(f != Z && f->type != T)
		et = f->type->etype;
	if(!typefd[et]) {
		diag(f, "fop: integer %O", o);
		return;
	}
	if(debug['M']) {
		if(t != Z && t->type != T)
			print("gop: %O %O-%s Z\n", o, f->op, tnames[et]);
		else
			print("gop: %O %O-%s %O-%s\n", o,
				f->op, tnames[et], t->op, tnames[t->type->etype]);
	}
	a = AGOK;
	switch(o) {

	case OASADD:
	case OADD:
		if(et == TFLOAT)
			a = AFADDF;
		else
		if(et == TDOUBLE || et == TVLONG) {
			a = AFADDD;
			if(pop)
				a = AFADDDP;
		}
		break;

	case OASSUB:
	case OSUB:
		if(et == TFLOAT) {
			a = AFSUBF;
			if(rev)
				a = AFSUBRF;
		} else
		if(et == TDOUBLE || et == TVLONG) {
			a = AFSUBD;
			if(pop)
				a = AFSUBDP;
			if(rev) {
				a = AFSUBRD;
				if(pop)
					a = AFSUBRDP;
			}
		}
		break;

	case OASMUL:
	case OMUL:
		if(et == TFLOAT)
			a = AFMULF;
		else
		if(et == TDOUBLE || et == TVLONG) {
			a = AFMULD;
			if(pop)
				a = AFMULDP;
		}
		break;

	case OASMOD:
	case OMOD:
	case OASDIV:
	case ODIV:
		if(et == TFLOAT) {
			a = AFDIVF;
			if(rev)
				a = AFDIVRF;
		} else
		if(et == TDOUBLE || et == TVLONG) {
			a = AFDIVD;
			if(pop)
				a = AFDIVDP;
			if(rev) {
				a = AFDIVRD;
				if(pop)
					a = AFDIVRDP;
			}
		}
		break;

	case OEQ:
	case ONE:
	case OLT:
	case OLE:
	case OGE:
	case OGT:
		pop += rev;
		if(et == TFLOAT) {
			a = AFCOMF;
			if(pop) {
				a = AFCOMFP;
				if(pop > 1)
					a = AGOK;
			}
		} else
		if(et == TDOUBLE || et == TVLONG) {
			a = AFCOMF;
			if(pop) {
				a = AFCOMDP;
				if(pop > 1)
					a = AFCOMDPP;
			}
		}
		gins(a, f, t);
		regalloc(&nod, &regnode, Z);
		if(nod.reg != D_AX) {
			regfree(&nod);
			nod.reg = D_AX;
			gins(APUSHL, &nod, Z);
			gins(AWAIT, Z, Z);
			gins(AFSTSW, Z, &nod);
			gins(ASAHF, Z, Z);
			gins(APOPL, Z, &nod);
		} else {
			gins(AWAIT, Z, Z);
			gins(AFSTSW, Z, &nod);
			gins(ASAHF, Z, Z);
			regfree(&nod);
		}
		switch(o) {
		case OEQ:	a = AJEQ; break;
		case ONE:	a = AJNE; break;
		case OLT:	a = AJCS; break;
		case OLE:	a = AJLS; break;
		case OGE:	a = AJCC; break;
		case OGT:	a = AJHI; break;
		}
		gins(a, Z, Z);
		return;
	}
	if(a == AGOK)
		diag(Z, "bad in gopcode %O", o);
	gins(a, f, t);
}

void
gopcode(int o, Type *ty, Node *f, Node *t)
{
	int a, et;

	et = TLONG;
	if(ty != T)
		et = ty->etype;
	if(typefd[et] && o != OADDR && o != OFUNC) {
		diag(f, "gop: float %O", o);
		return;
	}
	if(debug['M']) {
		if(f != Z && f->type != T)
			print("gop: %O %O[%s],", o, f->op, tnames[et]);
		else
			print("gop: %O Z,", o);
		if(t != Z && t->type != T)
			print("%O[%s]\n", t->op, tnames[t->type->etype]);
		else
			print("Z\n");
	}
	a = AGOK;
	switch(o) {
	case OCOM:
		a = ANOTL;
		if(et == TCHAR || et == TUCHAR)
			a = ANOTB;
		if(et == TSHORT || et == TUSHORT)
			a = ANOTW;
		break;

	case ONEG:
		a = ANEGL;
		if(et == TCHAR || et == TUCHAR)
			a = ANEGB;
		if(et == TSHORT || et == TUSHORT)
			a = ANEGW;
		break;

	case OADDR:
		a = ALEAL;
		break;

	case OASADD:
	case OADD:
		a = AADDL;
		if(et == TCHAR || et == TUCHAR)
			a = AADDB;
		if(et == TSHORT || et == TUSHORT)
			a = AADDW;
		break;

	case OASSUB:
	case OSUB:
		a = ASUBL;
		if(et == TCHAR || et == TUCHAR)
			a = ASUBB;
		if(et == TSHORT || et == TUSHORT)
			a = ASUBW;
		break;

	case OASOR:
	case OOR:
		a = AORL;
		if(et == TCHAR || et == TUCHAR)
			a = AORB;
		if(et == TSHORT || et == TUSHORT)
			a = AORW;
		break;

	case OASAND:
	case OAND:
		a = AANDL;
		if(et == TCHAR || et == TUCHAR)
			a = AANDB;
		if(et == TSHORT || et == TUSHORT)
			a = AANDW;
		break;

	case OASXOR:
	case OXOR:
		a = AXORL;
		if(et == TCHAR || et == TUCHAR)
			a = AXORB;
		if(et == TSHORT || et == TUSHORT)
			a = AXORW;
		break;

	case OASLSHR:
	case OLSHR:
		a = ASHRL;
		if(et == TCHAR || et == TUCHAR)
			a = ASHRB;
		if(et == TSHORT || et == TUSHORT)
			a = ASHRW;
		break;

	case OASASHR:
	case OASHR:
		a = ASARL;
		if(et == TCHAR || et == TUCHAR)
			a = ASARB;
		if(et == TSHORT || et == TUSHORT)
			a = ASARW;
		break;

	case OASASHL:
	case OASHL:
		a = ASALL;
		if(et == TCHAR || et == TUCHAR)
			a = ASALB;
		if(et == TSHORT || et == TUSHORT)
			a = ASALW;
		break;

	case OROTL:
		a = AROLL;
		if(et == TCHAR || et == TUCHAR)
			a = AROLB;
		if(et == TSHORT || et == TUSHORT)
			a = AROLW;
		break;

	case OFUNC:
		a = ACALL;
		break;

	case OASMUL:
	case OMUL:
		if(f->op == OREGISTER && t != Z && isreg(t, D_AX) && reg[D_DX] == 0)
			t = Z;
		a = AIMULL;
		break;

	case OASMOD:
	case OMOD:
	case OASDIV:
	case ODIV:
		a = AIDIVL;
		break;

	case OASLMUL:
	case OLMUL:
		a = AMULL;
		break;

	case OASLMOD:
	case OLMOD:
	case OASLDIV:
	case OLDIV:
		a = ADIVL;
		break;

	case OEQ:
	case ONE:
	case OLT:
	case OLE:
	case OGE:
	case OGT:
	case OLO:
	case OLS:
	case OHS:
	case OHI:
		a = ACMPL;
		if(et == TCHAR || et == TUCHAR)
			a = ACMPB;
		if(et == TSHORT || et == TUSHORT)
			a = ACMPW;
		gins(a, f, t);
		switch(o) {
		case OEQ:	a = AJEQ; break;
		case ONE:	a = AJNE; break;
		case OLT:	a = AJLT; break;
		case OLE:	a = AJLE; break;
		case OGE:	a = AJGE; break;
		case OGT:	a = AJGT; break;
		case OLO:	a = AJCS; break;
		case OLS:	a = AJLS; break;
		case OHS:	a = AJCC; break;
		case OHI:	a = AJHI; break;
		}
		gins(a, Z, Z);
		return;
	}
	if(a == AGOK)
		diag(Z, "bad in gopcode %O", o);
	gins(a, f, t);
}

int
samaddr(Node *f, Node *t)
{

	if(f->op != t->op)
		return 0;
	switch(f->op) {

	case OREGISTER:
		if(f->reg != t->reg)
			break;
		return 1;
	}
	return 0;
}

void
gbranch(int o)
{
	int a;

	a = AGOK;
	switch(o) {
	case ORETURN:
		a = ARET;
		break;
	case OGOTO:
		a = AJMP;
		break;
	}
	nextpc();
	if(a == AGOK) {
		diag(Z, "bad in gbranch %O",  o);
		nextpc();
	}
	p->as = a;
}

void
patch(Prog *op, int32 pc)
{
	op->to.offset = pc;
	op->to.type = D_BRANCH;
	op->to.u.branch = nil;
	op->pcond = nil;
}

void
gpseudo(int a, Sym *s, Node *n)
{

	nextpc();
	p->as = a;
	p->from.type = D_EXTERN;
	p->from.sym = linksym(s);

	switch(a) {
	case ATEXT:
		p->from.scale = textflag;
		textflag = 0;
		break;
	case AGLOBL:
		p->from.scale = s->dataflag;
		break;
	}

	if(s->class == CSTATIC)
		p->from.type = D_STATIC;
	naddr(n, &p->to);
	if(a == ADATA || a == AGLOBL)
		pc--;
}

void
gpcdata(int index, int value)
{
	Node n1;
	
	n1 = *nodconst(index);
	gins(APCDATA, &n1, nodconst(value));
}

void
gprefetch(Node *n)
{
	Node n1;
	
	if(strcmp(getgo386(), "sse2") != 0) // assume no prefetch on old machines
		return;

	regalloc(&n1, n, Z);
	gmove(n, &n1);
	n1.op = OINDREG;
	gins(APREFETCHNTA, &n1, Z);
	regfree(&n1);
}

int
sconst(Node *n)
{
	int32 v;

	if(n->op == OCONST && !typefd[n->type->etype]) {
		v = n->vconst;
		if(v >= -32766L && v < 32766L)
			return 1;
	}
	return 0;
}

int32
exreg(Type *t)
{
	int32 o;

	if(typechlp[t->etype]){
		if(exregoffset >= 32)
			return 0;
		o = exregoffset;
		exregoffset += 4;
		return o+1;	// +1 to avoid 0 == failure; naddr case OEXREG will -1.
	}

	return 0;
}

schar	ewidth[NTYPE] =
{
	-1,		/*[TXXX]*/
	SZ_CHAR,	/*[TCHAR]*/
	SZ_CHAR,	/*[TUCHAR]*/
	SZ_SHORT,	/*[TSHORT]*/
	SZ_SHORT,	/*[TUSHORT]*/
	SZ_INT,		/*[TINT]*/
	SZ_INT,		/*[TUINT]*/
	SZ_LONG,	/*[TLONG]*/
	SZ_LONG,	/*[TULONG]*/
	SZ_VLONG,	/*[TVLONG]*/
	SZ_VLONG,	/*[TUVLONG]*/
	SZ_FLOAT,	/*[TFLOAT]*/
	SZ_DOUBLE,	/*[TDOUBLE]*/
	SZ_IND,		/*[TIND]*/
	0,		/*[TFUNC]*/
	-1,		/*[TARRAY]*/
	0,		/*[TVOID]*/
	-1,		/*[TSTRUCT]*/
	-1,		/*[TUNION]*/
	SZ_INT,		/*[TENUM]*/
};
int32	ncast[NTYPE] =
{
	0,				/*[TXXX]*/
	BCHAR|BUCHAR,			/*[TCHAR]*/
	BCHAR|BUCHAR,			/*[TUCHAR]*/
	BSHORT|BUSHORT,			/*[TSHORT]*/
	BSHORT|BUSHORT,			/*[TUSHORT]*/
	BINT|BUINT|BLONG|BULONG|BIND,	/*[TINT]*/
	BINT|BUINT|BLONG|BULONG|BIND,	/*[TUINT]*/
	BINT|BUINT|BLONG|BULONG|BIND,	/*[TLONG]*/
	BINT|BUINT|BLONG|BULONG|BIND,	/*[TULONG]*/
	BVLONG|BUVLONG,			/*[TVLONG]*/
	BVLONG|BUVLONG,			/*[TUVLONG]*/
	BFLOAT,				/*[TFLOAT]*/
	BDOUBLE,			/*[TDOUBLE]*/
	BLONG|BULONG|BIND,		/*[TIND]*/
	0,				/*[TFUNC]*/
	0,				/*[TARRAY]*/
	0,				/*[TVOID]*/
	BSTRUCT,			/*[TSTRUCT]*/
	BUNION,				/*[TUNION]*/
	0,				/*[TENUM]*/
};
