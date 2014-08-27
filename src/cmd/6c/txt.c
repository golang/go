// Inferno utils/6c/txt.c
// http://code.google.com/p/inferno-os/source/browse/utils/6c/txt.c
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

int thechar = '6';
char *thestring = "amd64";

LinkArch	*thelinkarch = &linkamd64;

void
linkarchinit(void)
{
	if(strcmp(getgoarch(), "amd64p32") == 0)
		thelinkarch = &linkamd64p32;
}

void
ginit(void)
{
	int i;
	Type *t;

	dodefine("_64BITREG");
	if(ewidth[TIND] == 8)
		dodefine("_64BIT");
	listinit();
	nstring = 0;
	mnstring = 0;
	nrathole = 0;
	pc = 0;
	breakpc = -1;
	continpc = -1;
	cases = C;
	lastp = P;
	tfield = types[TINT];

	typeword = typechlvp;
	typecmplx = typesu;

	/* TO DO */
	memmove(typechlpv, typechlp, sizeof(typechlpv));
	typechlpv[TVLONG] = 1;
	typechlpv[TUVLONG] = 1;

	zprog.link = P;
	zprog.as = AGOK;
	zprog.from.type = D_NONE;
	zprog.from.index = D_NONE;
	zprog.from.scale = 0;
	zprog.to = zprog.from;

	lregnode.op = OREGISTER;
	lregnode.class = CEXREG;
	lregnode.reg = REGTMP;
	lregnode.complex = 0;
	lregnode.addable = 11;
	lregnode.type = types[TLONG];

	qregnode = lregnode;
	qregnode.type = types[TVLONG];

	constnode.op = OCONST;
	constnode.class = CXXX;
	constnode.complex = 0;
	constnode.addable = 20;
	constnode.type = types[TLONG];

	vconstnode = constnode;
	vconstnode.type = types[TVLONG];

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

	if(0)
		com64init();

	for(i=0; i<nelem(reg); i++) {
		reg[i] = 1;
		if(i >= D_AX && i <= D_R15 && i != D_SP)
			reg[i] = 0;
		if(i >= D_X0 && i <= D_X7)
			reg[i] = 0;
	}
	if(nacl) {
		reg[D_BP] = 1;
		reg[D_R15] = 1;
	}
}

void
gclean(void)
{
	int i;
	Sym *s;

	reg[D_SP]--;
	if(nacl) {
		reg[D_BP]--;
		reg[D_R15]--;
	}
	for(i=D_AX; i<=D_R15; i++)
		if(reg[i])
			diag(Z, "reg %R left allocated", i);
	for(i=D_X0; i<=D_X7; i++)
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
	for(i=D_AX; i<=D_R15; i++)
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
	if(typesu[n->type->etype]) {
		regaalloc(tn2, n);
		if(n->complex >= FNX) {
			sugen(*fnxp, tn2, n->type->width);
			(*fnxp)++;
		} else
			sugen(n, tn2, n->type->width);
		return;
	}
	if(REGARG >= 0 && curarg == 0 && typechlpv[n->type->etype]) {
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
nodgconst(vlong v, Type *t)
{
	if(!typev[t->etype])
		return nodconst((int32)v);
	vconstnode.vconst = v;
	return &vconstnode;
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
	int et;

	*n = qregnode;
	n->reg = r;
	if(nn != Z){
		et = nn->type->etype;
		if(!typefd[et] && nn->type->width <= SZ_LONG && 0)
			n->type = typeu[et]? types[TUINT]: types[TINT];
		else
			n->type = nn->type;
//print("nodreg %s [%s]\n", tnames[et], tnames[n->type->etype]);
		n->lineno = nn->lineno;
	}
	if(reg[r] == 0)
		return 0;
	if(nn != Z) {
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
		curarg = (curarg+7) & ~7;
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
		n->sym = slookup(".ret");
		n->complex = nodret->complex;
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
	case TVLONG:
	case TUVLONG:
	case TIND:
		if(o != Z && o->op == OREGISTER) {
			i = o->reg;
			if(i >= D_AX && i <= D_R15)
				goto out;
		}
		for(i=D_AX; i<=D_R15; i++)
			if(reg[i] == 0)
				goto out;
		diag(tn, "out of fixed registers");
		goto err;

	case TFLOAT:
	case TDOUBLE:
		if(o != Z && o->op == OREGISTER) {
			i = o->reg;
			if(i >= D_X0 && i <= D_X7)
				goto out;
		}
		for(i=D_X0; i<=D_X7; i++)
			if(reg[i] == 0)
				goto out;
		diag(tn, "out of float registers");
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
	reg[(uchar)REGARG]++;
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
		if(a->type >= D_AX && a->type <= D_R15)
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
		if(a->type >= D_AX && a->type <= D_R15)
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
		if(typev[n->type->etype] || (n->type->etype == TIND && ewidth[TIND] == 8))
			a->offset = n->vconst;
		else
			a->offset = convvtox(n->vconst, typeu[n->type->etype]? TULONG: TLONG);
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

void
gcmp(int op, Node *n, vlong val)
{
	Node *cn, nod;

	cn = nodgconst(val, n->type);
	if(!immconst(cn)){
		regalloc(&nod, n, Z);
		gmove(cn, &nod);
		gopcode(op, n->type, n, &nod);
		regfree(&nod);
	}else
		gopcode(op, n->type, n, cn);
}

#define	CASE(a,b)	((a<<8)|(b<<0))

void
gmove(Node *f, Node *t)
{
	int ft, tt, t64, a;
	Node nod, nod1, nod2, nod3;
	Prog *p1, *p2;

	ft = f->type->etype;
	tt = t->type->etype;
	if(ewidth[TIND] == 4) {
		if(ft == TIND)
			ft = TUINT;
		if(tt == TIND)
			tt = TUINT;
	}
	t64 = tt == TVLONG || tt == TUVLONG || tt == TIND;
	if(debug['M'])
		print("gop: %O %O[%s],%O[%s]\n", OAS,
			f->op, tnames[ft], t->op, tnames[tt]);
	if(typefd[ft] && f->op == OCONST) {
		/* TO DO: pick up special constants, possibly preloaded */
		if(f->fconst == 0.0){
			regalloc(&nod, t, t);
			gins(AXORPD, &nod, &nod);
			gmove(&nod, t);
			regfree(&nod);
			return;
		}
	}
/*
 * load
 */
	if(ft == TVLONG || ft == TUVLONG)
	if(f->op == OCONST)
	if(f->vconst > 0x7fffffffLL || f->vconst < -0x7fffffffLL)
	if(t->op != OREGISTER) {
		regalloc(&nod, f, Z);
		gmove(f, &nod);
		gmove(&nod, t);
		regfree(&nod);
		return;
	}

	if(f->op == ONAME || f->op == OINDREG ||
	   f->op == OIND || f->op == OINDEX)
	switch(ft) {
	case TCHAR:
		a = AMOVBLSX;
		if(t64)
			a = AMOVBQSX;
		goto ld;
	case TUCHAR:
		a = AMOVBLZX;
		if(t64)
			a = AMOVBQZX;
		goto ld;
	case TSHORT:
		a = AMOVWLSX;
		if(t64)
			a = AMOVWQSX;
		goto ld;
	case TUSHORT:
		a = AMOVWLZX;
		if(t64)
			a = AMOVWQZX;
		goto ld;
	case TINT:
	case TLONG:
		if(typefd[tt]) {
			regalloc(&nod, t, t);
			if(tt == TDOUBLE)
				a = ACVTSL2SD;
			else
				a = ACVTSL2SS;
			gins(a, f, &nod);
			gmove(&nod, t);
			regfree(&nod);
			return;
		}
		a = AMOVL;
		if(t64)
			a = AMOVLQSX;
		goto ld;
	case TUINT:
	case TULONG:
		a = AMOVL;
		if(t64)
			a = AMOVLQZX;	/* could probably use plain MOVL */
		goto ld;
	case TVLONG:
		if(typefd[tt]) {
			regalloc(&nod, t, t);
			if(tt == TDOUBLE)
				a = ACVTSQ2SD;
			else
				a = ACVTSQ2SS;
			gins(a, f, &nod);
			gmove(&nod, t);
			regfree(&nod);
			return;
		}
	case TUVLONG:
		a = AMOVQ;
		goto ld;
	case TIND:
		a = AMOVQ;
		if(ewidth[TIND] == 4)
			a = AMOVL;

	ld:
		regalloc(&nod, f, t);
		nod.type = t64? types[TVLONG]: types[TINT];
		gins(a, f, &nod);
		gmove(&nod, t);
		regfree(&nod);
		return;

	case TFLOAT:
		a = AMOVSS;
		goto fld;
	case TDOUBLE:
		a = AMOVSD;
	fld:
		regalloc(&nod, f, t);
		if(tt != TDOUBLE && tt != TFLOAT){	/* TO DO: why is this here */
			prtree(f, "odd tree");
			nod.type = t64? types[TVLONG]: types[TINT];
		}
		gins(a, f, &nod);
		gmove(&nod, t);
		regfree(&nod);
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
		a = AMOVL;	goto st;
	case TVLONG:
	case TUVLONG:
	case TIND:
		a = AMOVQ;	goto st;

	st:
		if(f->op == OCONST) {
			gins(a, f, t);
			return;
		}
	fst:
		regalloc(&nod, t, f);
		gmove(f, &nod);
		gins(a, &nod, t);
		regfree(&nod);
		return;

	case TFLOAT:
		a = AMOVSS;
		goto fst;
	case TDOUBLE:
		a = AMOVSD;
		goto fst;
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

	case CASE(	TCHAR,	TUCHAR):
	case CASE(	TUCHAR,	TUCHAR):
	case CASE(	TSHORT,	TUCHAR):
	case CASE(	TUSHORT,TUCHAR):
	case CASE(	TINT,	TUCHAR):
	case CASE(	TUINT,	TUCHAR):
	case CASE(	TLONG,	TUCHAR):
	case CASE(	TULONG,	TUCHAR):

	case CASE(	TSHORT,	TSHORT):
	case CASE(	TUSHORT,TSHORT):
	case CASE(	TINT,	TSHORT):
	case CASE(	TUINT,	TSHORT):
	case CASE(	TLONG,	TSHORT):
	case CASE(	TULONG,	TSHORT):

	case CASE(	TSHORT,	TUSHORT):
	case CASE(	TUSHORT,TUSHORT):
	case CASE(	TINT,	TUSHORT):
	case CASE(	TUINT,	TUSHORT):
	case CASE(	TLONG,	TUSHORT):
	case CASE(	TULONG,	TUSHORT):

	case CASE(	TINT,	TINT):
	case CASE(	TUINT,	TINT):
	case CASE(	TLONG,	TINT):
	case CASE(	TULONG,	TINT):

	case CASE(	TINT,	TUINT):
	case CASE(	TUINT,	TUINT):
	case CASE(	TLONG,	TUINT):
	case CASE(	TULONG,	TUINT):
 *****/
		a = AMOVL;
		break;

	case CASE(	TINT,	TIND):
	case CASE(	TINT,	TVLONG):
	case CASE(	TINT,	TUVLONG):
	case CASE(	TLONG,	TIND):
	case CASE(	TLONG,	TVLONG):
	case CASE(	TLONG,	TUVLONG):
		a = AMOVLQSX;
		if(f->op == OCONST) {
			f->vconst &= (uvlong)0xffffffffU;
			if(f->vconst & 0x80000000)
				f->vconst |= (vlong)0xffffffff << 32;
			a = AMOVQ;
		}
		break;

	case CASE(	TUINT,	TIND):
	case CASE(	TUINT,	TVLONG):
	case CASE(	TUINT,	TUVLONG):
	case CASE(	TULONG,	TVLONG):
	case CASE(	TULONG,	TUVLONG):
	case CASE(	TULONG,	TIND):
		a = AMOVLQZX;
		if(f->op == OCONST) {
			f->vconst &= (uvlong)0xffffffffU;
			a = AMOVQ;
		}
		break;
	
	case CASE(	TIND,	TCHAR):
	case CASE(	TIND,	TUCHAR):
	case CASE(	TIND,	TSHORT):
	case CASE(	TIND,	TUSHORT):
	case CASE(	TIND,	TINT):
	case CASE(	TIND,	TUINT):
	case CASE(	TIND,	TLONG):
	case CASE(	TIND,	TULONG):
	case CASE(	TVLONG,	TCHAR):
	case CASE(	TVLONG,	TUCHAR):
	case CASE(	TVLONG,	TSHORT):
	case CASE(	TVLONG,	TUSHORT):
	case CASE(	TVLONG,	TINT):
	case CASE(	TVLONG,	TUINT):
	case CASE(	TVLONG,	TLONG):
	case CASE(	TVLONG,	TULONG):
	case CASE(	TUVLONG,	TCHAR):
	case CASE(	TUVLONG,	TUCHAR):
	case CASE(	TUVLONG,	TSHORT):
	case CASE(	TUVLONG,	TUSHORT):
	case CASE(	TUVLONG,	TINT):
	case CASE(	TUVLONG,	TUINT):
	case CASE(	TUVLONG,	TLONG):
	case CASE(	TUVLONG,	TULONG):
		a = AMOVQL;
		if(f->op == OCONST) {
			f->vconst &= (int)0xffffffffU;
			a = AMOVL;
		}
		break;	

	case CASE(	TIND,	TIND):
	case CASE(	TIND,	TVLONG):
	case CASE(	TIND,	TUVLONG):
	case CASE(	TVLONG,	TIND):
	case CASE(	TVLONG,	TVLONG):
	case CASE(	TVLONG,	TUVLONG):
	case CASE(	TUVLONG,	TIND):
	case CASE(	TUVLONG,	TVLONG):
	case CASE(	TUVLONG,	TUVLONG):
		a = AMOVQ;
		break;

	case CASE(	TSHORT,	TINT):
	case CASE(	TSHORT,	TUINT):
	case CASE(	TSHORT,	TLONG):
	case CASE(	TSHORT,	TULONG):
		a = AMOVWLSX;
		if(f->op == OCONST) {
			f->vconst &= 0xffff;
			if(f->vconst & 0x8000)
				f->vconst |= 0xffff0000;
			a = AMOVL;
		}
		break;

	case CASE(	TSHORT,	TVLONG):
	case CASE(	TSHORT,	TUVLONG):
	case CASE(	TSHORT,	TIND):
		a = AMOVWQSX;
		if(f->op == OCONST) {
			f->vconst &= 0xffff;
			if(f->vconst & 0x8000){
				f->vconst |= 0xffff0000;
				f->vconst |= (vlong)~0 << 32;
			}
			a = AMOVL;
		}
		break;

	case CASE(	TUSHORT,TINT):
	case CASE(	TUSHORT,TUINT):
	case CASE(	TUSHORT,TLONG):
	case CASE(	TUSHORT,TULONG):
		a = AMOVWLZX;
		if(f->op == OCONST) {
			f->vconst &= 0xffff;
			a = AMOVL;
		}
		break;

	case CASE(	TUSHORT,TVLONG):
	case CASE(	TUSHORT,TUVLONG):
	case CASE(	TUSHORT,TIND):
		a = AMOVWQZX;
		if(f->op == OCONST) {
			f->vconst &= 0xffff;
			a = AMOVL;	/* MOVL also zero-extends to 64 bits */
		}
		break;

	case CASE(	TCHAR,	TSHORT):
	case CASE(	TCHAR,	TUSHORT):
	case CASE(	TCHAR,	TINT):
	case CASE(	TCHAR,	TUINT):
	case CASE(	TCHAR,	TLONG):
	case CASE(	TCHAR,	TULONG):
		a = AMOVBLSX;
		if(f->op == OCONST) {
			f->vconst &= 0xff;
			if(f->vconst & 0x80)
				f->vconst |= 0xffffff00;
			a = AMOVL;
		}
		break;

	case CASE(	TCHAR,	TVLONG):
	case CASE(	TCHAR,	TUVLONG):
	case CASE(	TCHAR,	TIND):
		a = AMOVBQSX;
		if(f->op == OCONST) {
			f->vconst &= 0xff;
			if(f->vconst & 0x80){
				f->vconst |= 0xffffff00;
				f->vconst |= (vlong)~0 << 32;
			}
			a = AMOVQ;
		}
		break;

	case CASE(	TUCHAR,	TSHORT):
	case CASE(	TUCHAR,	TUSHORT):
	case CASE(	TUCHAR,	TINT):
	case CASE(	TUCHAR,	TUINT):
	case CASE(	TUCHAR,	TLONG):
	case CASE(	TUCHAR,	TULONG):
		a = AMOVBLZX;
		if(f->op == OCONST) {
			f->vconst &= 0xff;
			a = AMOVL;
		}
		break;

	case CASE(	TUCHAR,	TVLONG):
	case CASE(	TUCHAR,	TUVLONG):
	case CASE(	TUCHAR,	TIND):
		a = AMOVBQZX;
		if(f->op == OCONST) {
			f->vconst &= 0xff;
			a = AMOVL;	/* zero-extends to 64-bits */
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
	case CASE(	TFLOAT,	TVLONG):
	case CASE(	TFLOAT,	TUVLONG):
	case CASE(	TFLOAT,	TIND):

	case CASE(	TDOUBLE,TCHAR):
	case CASE(	TDOUBLE,TUCHAR):
	case CASE(	TDOUBLE,TSHORT):
	case CASE(	TDOUBLE,TUSHORT):
	case CASE(	TDOUBLE,TINT):
	case CASE(	TDOUBLE,TUINT):
	case CASE(	TDOUBLE,TLONG):
	case CASE(	TDOUBLE,TULONG):
	case CASE(	TDOUBLE,TVLONG):
	case CASE(	TDOUBLE,TUVLONG):
	case CASE(	TDOUBLE,TIND):
		regalloc(&nod, t, Z);
		if(ewidth[tt] == SZ_VLONG || typeu[tt] && ewidth[tt] == SZ_INT){
			if(ft == TFLOAT)
				a = ACVTTSS2SQ;
			else
				a = ACVTTSD2SQ;
		}else{
			if(ft == TFLOAT)
				a = ACVTTSS2SL;
			else
				a = ACVTTSD2SL;
		}
		gins(a, f, &nod);
		gmove(&nod, t);
		regfree(&nod);
		return;

/*
 * uvlong to float
 */
	case CASE(	TUVLONG,	TDOUBLE):
	case CASE(	TUVLONG,	TFLOAT):
		a = ACVTSQ2SS;
		if(tt == TDOUBLE)
			a = ACVTSQ2SD;
		regalloc(&nod, f, f);
		gmove(f, &nod);
		regalloc(&nod1, t, t);
		gins(ACMPQ, &nod, nodconst(0));
		gins(AJLT, Z, Z);
		p1 = p;
		gins(a, &nod, &nod1);
		gins(AJMP, Z, Z);
		p2 = p;
		patch(p1, pc);
		regalloc(&nod2, f, Z);
		regalloc(&nod3, f, Z);
		gmove(&nod, &nod2);
		gins(ASHRQ, nodconst(1), &nod2);
		gmove(&nod, &nod3);
		gins(AANDL, nodconst(1), &nod3);
		gins(AORQ, &nod3, &nod2);
		gins(a, &nod2, &nod1);
		gins(tt == TDOUBLE? AADDSD: AADDSS, &nod1, &nod1);
		regfree(&nod2);
		regfree(&nod3);
		patch(p2, pc);
		regfree(&nod);
		regfree(&nod1);
		return;

	case CASE(	TULONG,	TDOUBLE):
	case CASE(	TUINT,	TDOUBLE):
	case CASE(	TULONG,	TFLOAT):
	case CASE(	TUINT,	TFLOAT):
		a = ACVTSQ2SS;
		if(tt == TDOUBLE)
			a = ACVTSQ2SD;
		regalloc(&nod, f, f);
		gins(AMOVLQZX, f, &nod);
		regalloc(&nod1, t, t);
		gins(a, &nod, &nod1);
		gmove(&nod1, t);
		regfree(&nod);
		regfree(&nod1);
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
	case	CASE(	TVLONG,	TFLOAT):
	case CASE(	TIND,	TFLOAT):

	case CASE(	TCHAR,	TDOUBLE):
	case CASE(	TUCHAR,	TDOUBLE):
	case CASE(	TSHORT,	TDOUBLE):
	case CASE(	TUSHORT,TDOUBLE):
	case CASE(	TINT,	TDOUBLE):
	case CASE(	TLONG,	TDOUBLE):
	case CASE(	TVLONG,	TDOUBLE):
	case CASE(	TIND,	TDOUBLE):
		regalloc(&nod, t, t);
		if(ewidth[ft] == SZ_VLONG){
			if(tt == TFLOAT)
				a = ACVTSQ2SS;
			else
				a = ACVTSQ2SD;
		}else{
			if(tt == TFLOAT)
				a = ACVTSL2SS;
			else
				a = ACVTSL2SD;
		}
		gins(a, f, &nod);
		gmove(&nod, t);
		regfree(&nod);
		return;

/*
 * float to float
 */
	case CASE(	TFLOAT,	TFLOAT):
		a = AMOVSS;
		break;
	case CASE(	TDOUBLE,TFLOAT):
		a = ACVTSD2SS;
		break;
	case CASE(	TFLOAT,	TDOUBLE):
		a = ACVTSS2SD;
		break;
	case CASE(	TDOUBLE,TDOUBLE):
		a = AMOVSD;
		break;
	}
	if(a == AMOVQ || a == AMOVSD || a == AMOVSS || a == AMOVL && ewidth[ft] == ewidth[tt])	/* TO DO: check AMOVL */
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

	regalloc(&nod, &qregnode, Z);
	v = constnode.vconst;
	cgen(n->right, &nod);
	idx.ptr = D_NONE;
	if(n->left->op == OCONST)
		idx.ptr = D_CONST;
	else if(n->left->op == OREGISTER)
		idx.ptr = n->left->reg;
	else if(n->left->op != OADDR) {
		reg[D_BP]++;	// can't be used as a base
		regalloc(&nod1, &qregnode, Z);
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
gopcode(int o, Type *ty, Node *f, Node *t)
{
	int a, et;

	et = TLONG;
	if(ty != T)
		et = ty->etype;
	if(et == TIND && ewidth[TIND] == 4)
		et = TUINT;
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
		if(et == TVLONG || et == TUVLONG || et == TIND)
			a = ANOTQ;
		break;

	case ONEG:
		a = ANEGL;
		if(et == TCHAR || et == TUCHAR)
			a = ANEGB;
		if(et == TSHORT || et == TUSHORT)
			a = ANEGW;
		if(et == TVLONG || et == TUVLONG || et == TIND)
			a = ANEGQ;
		break;

	case OADDR:
		a = ALEAQ;
		break;

	case OASADD:
	case OADD:
		a = AADDL;
		if(et == TCHAR || et == TUCHAR)
			a = AADDB;
		if(et == TSHORT || et == TUSHORT)
			a = AADDW;
		if(et == TVLONG || et == TUVLONG || et == TIND)
			a = AADDQ;
		if(et == TFLOAT)
			a = AADDSS;
		if(et == TDOUBLE)
			a = AADDSD;
		break;

	case OASSUB:
	case OSUB:
		a = ASUBL;
		if(et == TCHAR || et == TUCHAR)
			a = ASUBB;
		if(et == TSHORT || et == TUSHORT)
			a = ASUBW;
		if(et == TVLONG || et == TUVLONG || et == TIND)
			a = ASUBQ;
		if(et == TFLOAT)
			a = ASUBSS;
		if(et == TDOUBLE)
			a = ASUBSD;
		break;

	case OASOR:
	case OOR:
		a = AORL;
		if(et == TCHAR || et == TUCHAR)
			a = AORB;
		if(et == TSHORT || et == TUSHORT)
			a = AORW;
		if(et == TVLONG || et == TUVLONG || et == TIND)
			a = AORQ;
		break;

	case OASAND:
	case OAND:
		a = AANDL;
		if(et == TCHAR || et == TUCHAR)
			a = AANDB;
		if(et == TSHORT || et == TUSHORT)
			a = AANDW;
		if(et == TVLONG || et == TUVLONG || et == TIND)
			a = AANDQ;
		break;

	case OASXOR:
	case OXOR:
		a = AXORL;
		if(et == TCHAR || et == TUCHAR)
			a = AXORB;
		if(et == TSHORT || et == TUSHORT)
			a = AXORW;
		if(et == TVLONG || et == TUVLONG || et == TIND)
			a = AXORQ;
		break;

	case OASLSHR:
	case OLSHR:
		a = ASHRL;
		if(et == TCHAR || et == TUCHAR)
			a = ASHRB;
		if(et == TSHORT || et == TUSHORT)
			a = ASHRW;
		if(et == TVLONG || et == TUVLONG || et == TIND)
			a = ASHRQ;
		break;

	case OASASHR:
	case OASHR:
		a = ASARL;
		if(et == TCHAR || et == TUCHAR)
			a = ASARB;
		if(et == TSHORT || et == TUSHORT)
			a = ASARW;
		if(et == TVLONG || et == TUVLONG || et == TIND)
			a = ASARQ;
		break;

	case OASASHL:
	case OASHL:
		a = ASALL;
		if(et == TCHAR || et == TUCHAR)
			a = ASALB;
		if(et == TSHORT || et == TUSHORT)
			a = ASALW;
		if(et == TVLONG || et == TUVLONG || et == TIND)
			a = ASALQ;
		break;

	case OROTL:
		a = AROLL;
		if(et == TCHAR || et == TUCHAR)
			a = AROLB;
		if(et == TSHORT || et == TUSHORT)
			a = AROLW;
		if(et == TVLONG || et == TUVLONG || et == TIND)
			a = AROLQ;
		break;

	case OFUNC:
		a = ACALL;
		break;

	case OASMUL:
	case OMUL:
		if(f->op == OREGISTER && t != Z && isreg(t, D_AX) && reg[D_DX] == 0)
			t = Z;
		a = AIMULL;
		if(et == TVLONG || et == TUVLONG || et == TIND)
			a = AIMULQ;
		if(et == TFLOAT)
			a = AMULSS;
		if(et == TDOUBLE)
			a = AMULSD;
		break;

	case OASMOD:
	case OMOD:
	case OASDIV:
	case ODIV:
		a = AIDIVL;
		if(et == TVLONG || et == TUVLONG || et == TIND)
			a = AIDIVQ;
		if(et == TFLOAT)
			a = ADIVSS;
		if(et == TDOUBLE)
			a = ADIVSD;
		break;

	case OASLMUL:
	case OLMUL:
		a = AMULL;
		if(et == TVLONG || et == TUVLONG || et == TIND)
			a = AMULQ;
		break;

	case OASLMOD:
	case OLMOD:
	case OASLDIV:
	case OLDIV:
		a = ADIVL;
		if(et == TVLONG || et == TUVLONG || et == TIND)
			a = ADIVQ;
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
		if(et == TVLONG || et == TUVLONG || et == TIND)
			a = ACMPQ;
		if(et == TFLOAT)
			a = AUCOMISS;
		if(et == TDOUBLE)
			a = AUCOMISD;
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
	return f->op == OREGISTER && t->op == OREGISTER && f->reg == t->reg;
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

	if(typechlpv[t->etype]) {
		if(exregoffset >= 64)
			return 0;
		o = exregoffset;
		exregoffset += ewidth[TIND];
		return o+1;	// +1 to avoid 0 == failure; naddr's case OEXREG will subtract 1.
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
	BINT|BUINT|BLONG|BULONG,	/*[TINT]*/
	BINT|BUINT|BLONG|BULONG,	/*[TUINT]*/
	BINT|BUINT|BLONG|BULONG,	/*[TLONG]*/
	BINT|BUINT|BLONG|BULONG,	/*[TULONG]*/
	BVLONG|BUVLONG|BIND,			/*[TVLONG]*/
	BVLONG|BUVLONG|BIND,			/*[TUVLONG]*/
	BFLOAT,				/*[TFLOAT]*/
	BDOUBLE,			/*[TDOUBLE]*/
	BVLONG|BUVLONG|BIND,		/*[TIND]*/
	0,				/*[TFUNC]*/
	0,				/*[TARRAY]*/
	0,				/*[TVOID]*/
	BSTRUCT,			/*[TSTRUCT]*/
	BUNION,				/*[TUNION]*/
	0,				/*[TENUM]*/
};
