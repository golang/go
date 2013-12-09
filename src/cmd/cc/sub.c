// Inferno utils/cc/sub.c
// http://code.google.com/p/inferno-os/source/browse/utils/cc/sub.c
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

#include	<u.h>
#include	"cc.h"

Node*
new(int t, Node *l, Node *r)
{
	Node *n;

	n = alloc(sizeof(*n));
	n->op = t;
	n->left = l;
	n->right = r;
	if(l && t != OGOTO)
		n->lineno = l->lineno;
	else if(r)
		n->lineno = r->lineno;
	else
		n->lineno = lineno;
	newflag = 1;
	return n;
}

Node*
new1(int o, Node *l, Node *r)
{
	Node *n;

	n = new(o, l, r);
	n->lineno = nearln;
	return n;
}

void
prtree(Node *n, char *s)
{

	print(" == %s ==\n", s);
	prtree1(n, 0, 0);
	print("\n");
}

void
prtree1(Node *n, int d, int f)
{
	int i;

	if(f)
	for(i=0; i<d; i++)
		print("   ");
	if(n == Z) {
		print("Z\n");
		return;
	}
	if(n->op == OLIST) {
		prtree1(n->left, d, 0);
		prtree1(n->right, d, 1);
		return;
	}
	d++;
	print("%O", n->op);
	i = 3;
	switch(n->op)
	{
	case ONAME:
		print(" \"%F\"", n);
		print(" %d", n->xoffset);
		i = 0;
		break;

	case OINDREG:
		print(" %d(R%d)", n->xoffset, n->reg);
		i = 0;
		break;

	case OREGISTER:
		if(n->xoffset)
			print(" %d+R%d", n->xoffset, n->reg);
		else
			print(" R%d", n->reg);
		i = 0;
		break;

	case OSTRING:
		print(" \"%s\"", n->cstring);
		i = 0;
		break;

	case OLSTRING:
		if(sizeof(TRune) == sizeof(Rune))
			print(" \"%S\"", (Rune*)n->rstring);
		else
			print(" \"...\"");
		i = 0;
		break;

	case ODOT:
	case OELEM:
		print(" \"%F\"", n);
		break;

	case OCONST:
		if(typefd[n->type->etype])
			print(" \"%.8e\"", n->fconst);
		else
			print(" \"%lld\"", n->vconst);
		i = 0;
		break;
	}
	if(n->addable != 0)
		print(" <%d>", n->addable);
	if(n->type != T)
		print(" %T", n->type);
	if(n->complex != 0)
		print(" (%d)", n->complex);
	print(" %L\n", n->lineno);
	if(i & 2)
		prtree1(n->left, d, 1);
	if(i & 1)
		prtree1(n->right, d, 1);
}

Type*
typ(int et, Type *d)
{
	Type *t;

	t = alloc(sizeof(*t));
	t->etype = et;
	t->link = d;
	t->down = T;
	t->sym = S;
	if(et < NTYPE)
		t->width = ewidth[et];
	else
		t->width = -1; // for TDOT or TOLD in prototype
	t->offset = 0;
	t->shift = 0;
	t->nbits = 0;
	t->garb = 0;
	return t;
}

Type*
copytyp(Type *t)
{
	Type *nt;

	nt = typ(TXXX, T);
	*nt = *t;
	return nt;
}

Type*
garbt(Type *t, int32 b)
{
	Type *t1;

	if(b & BGARB) {
		t1 = copytyp(t);
		t1->garb = simpleg(b);
		return t1;
	}
	return t;
}

int
simpleg(int32 b)
{

	b &= BGARB;
	switch(b) {
	case BCONSTNT:
		return GCONSTNT;
	case BVOLATILE:
		return GVOLATILE;
	case BVOLATILE|BCONSTNT:
		return GCONSTNT|GVOLATILE;
	}
	return GXXX;
}

int
simplec(int32 b)
{

	b &= BCLASS;
	switch(b) {
	case 0:
	case BREGISTER:
		return CXXX;
	case BAUTO:
	case BAUTO|BREGISTER:
		return CAUTO;
	case BEXTERN:
		return CEXTERN;
	case BEXTERN|BREGISTER:
		return CEXREG;
	case BSTATIC:
		return CSTATIC;
	case BTYPEDEF:
		return CTYPEDEF;
	case BTYPESTR:
		return CTYPESTR;
	}
	diag(Z, "illegal combination of classes %Q", b);
	return CXXX;
}

Type*
simplet(int32 b)
{

	b &= ~BCLASS & ~BGARB;
	switch(b) {
	case BCHAR:
	case BCHAR|BSIGNED:
		return types[TCHAR];

	case BCHAR|BUNSIGNED:
		return types[TUCHAR];

	case BSHORT:
	case BSHORT|BINT:
	case BSHORT|BSIGNED:
	case BSHORT|BINT|BSIGNED:
		return types[TSHORT];

	case BUNSIGNED|BSHORT:
	case BUNSIGNED|BSHORT|BINT:
		return types[TUSHORT];

	case 0:
	case BINT:
	case BINT|BSIGNED:
	case BSIGNED:
		return types[TINT];

	case BUNSIGNED:
	case BUNSIGNED|BINT:
		return types[TUINT];

	case BLONG:
	case BLONG|BINT:
	case BLONG|BSIGNED:
	case BLONG|BINT|BSIGNED:
		return types[TLONG];

	case BUNSIGNED|BLONG:
	case BUNSIGNED|BLONG|BINT:
		return types[TULONG];

	case BVLONG|BLONG:
	case BVLONG|BLONG|BINT:
	case BVLONG|BLONG|BSIGNED:
	case BVLONG|BLONG|BINT|BSIGNED:
		return types[TVLONG];

	case BVLONG|BLONG|BUNSIGNED:
	case BVLONG|BLONG|BINT|BUNSIGNED:
		return types[TUVLONG];

	case BFLOAT:
		return types[TFLOAT];

	case BDOUBLE:
	case BDOUBLE|BLONG:
	case BFLOAT|BLONG:
		return types[TDOUBLE];

	case BVOID:
		return types[TVOID];
	}

	diag(Z, "illegal combination of types %Q", b);
	return types[TINT];
}

int
stcompat(Node *n, Type *t1, Type *t2, int32 ttab[])
{
	int i;
	uint32 b;

	i = 0;
	if(t2 != T)
		i = t2->etype;
	b = 1L << i;
	i = 0;
	if(t1 != T)
		i = t1->etype;
	if(b & ttab[i]) {
		if(ttab == tasign)
			if(b == BSTRUCT || b == BUNION)
				if(!sametype(t1, t2))
					return 1;
		if(n->op != OCAST)
		 	if(b == BIND && i == TIND)
				if(!sametype(t1, t2))
					return 1;
		return 0;
	}
	return 1;
}

int
tcompat(Node *n, Type *t1, Type *t2, int32 ttab[])
{

	if(stcompat(n, t1, t2, ttab)) {
		if(t1 == T)
			diag(n, "incompatible type: \"%T\" for op \"%O\"",
				t2, n->op);
		else
			diag(n, "incompatible types: \"%T\" and \"%T\" for op \"%O\"",
				t1, t2, n->op);
		return 1;
	}
	return 0;
}

void
makedot(Node *n, Type *t, int32 o)
{
	Node *n1, *n2;

	if(t->nbits) {
		n1 = new(OXXX, Z, Z);
		*n1 = *n;
		n->op = OBIT;
		n->left = n1;
		n->right = Z;
		n->type = t;
		n->addable = n1->left->addable;
		n = n1;
	}
	n->addable = n->left->addable;
	if(n->addable == 0) {
		n1 = new1(OCONST, Z, Z);
		n1->vconst = o;
		n1->type = types[TLONG];
		n->right = n1;
		n->type = t;
		return;
	}
	n->left->type = t;
	if(o == 0) {
		*n = *n->left;
		return;
	}
	n->type = t;
	n1 = new1(OCONST, Z, Z);
	n1->vconst = o;
	t = typ(TIND, t);
	t->width = types[TIND]->width;
	n1->type = t;

	n2 = new1(OADDR, n->left, Z);
	n2->type = t;

	n1 = new1(OADD, n1, n2);
	n1->type = t;

	n->op = OIND;
	n->left = n1;
	n->right = Z;
}

Type*
dotsearch(Sym *s, Type *t, Node *n, int32 *off)
{
	Type *t1, *xt, *rt;

	xt = T;

	/*
	 * look it up by name
	 */
	for(t1 = t; t1 != T; t1 = t1->down)
		if(t1->sym == s) {
			if(xt != T)
				goto ambig;
			xt = t1;
		}

	/*
	 * look it up by type
	 */
	if(s->class == CTYPEDEF || s->class == CTYPESTR)
		for(t1 = t; t1 != T; t1 = t1->down)
			if(t1->sym == S && typesu[t1->etype])
				if(sametype(s->type, t1)) {
					if(xt != T)
						goto ambig;
					xt = t1;
				}
	if(xt != T) {
		*off = xt->offset;
		return xt;
	}

	/*
	 * look it up in unnamed substructures
	 */
	for(t1 = t; t1 != T; t1 = t1->down)
		if(t1->sym == S && typesu[t1->etype]){
			rt = dotsearch(s, t1->link, n, off);
			if(rt != T) {
				if(xt != T)
					goto ambig;
				xt = rt;
				*off += t1->offset;
			}
		}
	return xt;

ambig:
	diag(n, "ambiguous structure element: %s", s->name);
	return xt;
}

int32
dotoffset(Type *st, Type *lt, Node *n)
{
	Type *t;
	Sym *g;
	int32 o, o1;

	o = -1;
	/*
	 * first try matching at the top level
	 * for matching tag names
	 */
	g = st->tag;
	if(g != S)
		for(t=lt->link; t!=T; t=t->down)
			if(t->sym == S)
				if(g == t->tag) {
					if(o >= 0)
						goto ambig;
					o = t->offset;
				}
	if(o >= 0)
		return o;

	/*
	 * second try matching at the top level
	 * for similar types
	 */
	for(t=lt->link; t!=T; t=t->down)
		if(t->sym == S)
			if(sametype(st, t)) {
				if(o >= 0)
					goto ambig;
				o = t->offset;
			}
	if(o >= 0)
		return o;

	/*
	 * last try matching sub-levels
	 */
	for(t=lt->link; t!=T; t=t->down)
		if(t->sym == S)
		if(typesu[t->etype]) {
			o1 = dotoffset(st, t, n);
			if(o1 >= 0) {
				if(o >= 0)
					goto ambig;
				o = o1 + t->offset;
			}
		}
	return o;

ambig:
	diag(n, "ambiguous unnamed structure element");
	return o;
}

/*
 * look into tree for floating point constant expressions
 */
int
allfloat(Node *n, int flag)
{

	if(n != Z) {
		if(n->type->etype != TDOUBLE)
			return 1;
		switch(n->op) {
		case OCONST:
			if(flag)
				n->type = types[TFLOAT];
			return 1;
		case OADD:	/* no need to get more exotic than this */
		case OSUB:
		case OMUL:
		case ODIV:
			if(!allfloat(n->right, flag))
				break;
		case OCAST:
			if(!allfloat(n->left, flag))
				break;
			if(flag)
				n->type = types[TFLOAT];
			return 1;
		}
	}
	return 0;
}

void
constas(Node *n, Type *il, Type *ir)
{
	Type *l, *r;

	l = il;
	r = ir;

	if(l == T)
		return;
	if(l->garb & GCONSTNT) {
		warn(n, "assignment to a constant type (%T)", il);
		return;
	}
	if(r == T)
		return;
	for(;;) {
		if(l->etype != TIND || r->etype != TIND)
			break;
		l = l->link;
		r = r->link;
		if(l == T || r == T)
			break;
		if(r->garb & GCONSTNT)
			if(!(l->garb & GCONSTNT)) {
				warn(n, "assignment of a constant pointer type (%T)", ir);
				break;
			}
	}
}

void
typeext1(Type *st, Node *l)
{
	if(st->etype == TFLOAT && allfloat(l, 0))
		allfloat(l, 1);
}

void
typeext(Type *st, Node *l)
{
	Type *lt;
	Node *n1, *n2;
	int32 o;

	lt = l->type;
	if(lt == T)
		return;
	if(st->etype == TIND && vconst(l) == 0) {
		l->type = st;
		l->vconst = 0;
		return;
	}
	typeext1(st, l);

	/*
	 * extension of C
	 * if assign of struct containing unnamed sub-struct
	 * to type of sub-struct, insert the DOT.
	 * if assign of *struct containing unnamed substruct
	 * to type of *sub-struct, insert the add-offset
	 */
	if(typesu[st->etype] && typesu[lt->etype]) {
		o = dotoffset(st, lt, l);
		if(o >= 0) {
			n1 = new1(OXXX, Z, Z);
			*n1 = *l;
			l->op = ODOT;
			l->left = n1;
			l->right = Z;
			makedot(l, st, o);
		}
		return;
	}
	if(st->etype == TIND && typesu[st->link->etype])
	if(lt->etype == TIND && typesu[lt->link->etype]) {
		o = dotoffset(st->link, lt->link, l);
		if(o >= 0) {
			l->type = st;
			if(o == 0)
				return;
			n1 = new1(OXXX, Z, Z);
			*n1 = *l;
			n2 = new1(OCONST, Z, Z);
			n2->vconst = o;
			n2->type = st;
			l->op = OADD;
			l->left = n1;
			l->right = n2;
		}
		return;
	}
}

/*
 * a cast that generates no code
 * (same size move)
 */
int
nocast(Type *t1, Type *t2)
{
	int i, b;

	if(t1->nbits)
		return 0;
	i = 0;
	if(t2 != T)
		i = t2->etype;
	b = 1<<i;
	i = 0;
	if(t1 != T)
		i = t1->etype;
	if(b & ncast[i])
		return 1;
	return 0;
}

/*
 * a cast that has a noop semantic
 * (small to large, convert)
 */
int
nilcast(Type *t1, Type *t2)
{
	int et1, et2;

	if(t1 == T)
		return 0;
	if(t1->nbits)
		return 0;
	if(t2 == T)
		return 0;
	et1 = t1->etype;
	et2 = t2->etype;
	if(et1 == et2)
		return 1;
	if(typefd[et1] && typefd[et2]) {
		if(ewidth[et1] < ewidth[et2])
			return 1;
		return 0;
	}
	if(typechlp[et1] && typechlp[et2]) {
		if(ewidth[et1] < ewidth[et2])
			return 1;
		return 0;
	}
	return 0;
}

/*
 * "the usual arithmetic conversions are performed"
 */
void
arith(Node *n, int f)
{
	Type *t1, *t2;
	int i, j, k;
	Node *n1;
	int32 w;

	t1 = n->left->type;
	if(n->right == Z)
		t2 = t1;
	else
		t2 = n->right->type;
	i = TXXX;
	if(t1 != T)
		i = t1->etype;
	j = TXXX;
	if(t2 != T)
		j = t2->etype;
	k = tab[i][j];
	if(k == TIND) {
		if(i == TIND)
			n->type = t1;
		else
		if(j == TIND)
			n->type = t2;
	} else {
		/* convert up to at least int */
		if(f == 1)
		while(k < TINT)
			k += 2;
		n->type = types[k];
	}
	if(n->op == OSUB)
	if(i == TIND && j == TIND) {
		w = n->right->type->link->width;
		if(w < 1 || n->left->type->link == T || n->left->type->link->width < 1)
			goto bad;
		n->type = types[ewidth[TIND] <= ewidth[TLONG]? TLONG: TVLONG];
		if(0 && ewidth[TIND] > ewidth[TLONG]){
			n1 = new1(OXXX, Z, Z);
			*n1 = *n;
			n->op = OCAST;
			n->left = n1;
			n->right = Z;
			n->type = types[TLONG];
		}
		if(w > 1) {
			n1 = new1(OXXX, Z, Z);
			*n1 = *n;
			n->op = ODIV;
			n->left = n1;
			n1 = new1(OCONST, Z, Z);
			n1->vconst = w;
			n1->type = n->type;
			n->right = n1;
			w = vlog(n1);
			if(w >= 0) {
				n->op = OASHR;
				n1->vconst = w;
			}
		}
		return;
	}
	if(!sametype(n->type, n->left->type)) {
		n->left = new1(OCAST, n->left, Z);
		n->left->type = n->type;
		if(n->type->etype == TIND) {
			w = n->type->link->width;
			if(w < 1) {
				snap(n->type->link);
				w = n->type->link->width;
				if(w < 1)
					goto bad;
			}
			if(w > 1) {
				n1 = new1(OCONST, Z, Z);
				n1->vconst = w;
				n1->type = n->type;
				n->left = new1(OMUL, n->left, n1);
				n->left->type = n->type;
			}
		}
	}
	if(n->right != Z)
	if(!sametype(n->type, n->right->type)) {
		n->right = new1(OCAST, n->right, Z);
		n->right->type = n->type;
		if(n->type->etype == TIND) {
			w = n->type->link->width;
			if(w < 1) {
				snap(n->type->link);
				w = n->type->link->width;
				if(w < 1)
					goto bad;
			}
			if(w != 1) {
				n1 = new1(OCONST, Z, Z);
				n1->vconst = w;
				n1->type = n->type;
				n->right = new1(OMUL, n->right, n1);
				n->right->type = n->type;
			}
		}
	}
	return;
bad:
	diag(n, "pointer addition not fully declared: %T", n->type->link);
}

/*
 * try to rewrite shift & mask
 */
void
simplifyshift(Node *n)
{
	uint32 c3;
	int o, s1, s2, c1, c2;

	if(!typechlp[n->type->etype])
		return;
	switch(n->op) {
	default:
		return;
	case OASHL:
		s1 = 0;
		break;
	case OLSHR:
		s1 = 1;
		break;
	case OASHR:
		s1 = 2;
		break;
	}
	if(n->right->op != OCONST)
		return;
	if(n->left->op != OAND)
		return;
	if(n->left->right->op != OCONST)
		return;
	switch(n->left->left->op) {
	default:
		return;
	case OASHL:
		s2 = 0;
		break;
	case OLSHR:
		s2 = 1;
		break;
	case OASHR:
		s2 = 2;
		break;
	}
	if(n->left->left->right->op != OCONST)
		return;

	c1 = n->right->vconst;
	c2 = n->left->left->right->vconst;
	c3 = n->left->right->vconst;

	o = n->op;
	switch((s1<<3)|s2) {
	case 000:	/* (((e <<u c2) & c3) <<u c1) */
		c3 >>= c2;
		c1 += c2;
		if(c1 >= 32)
			break;
		goto rewrite1;

	case 002:	/* (((e >>s c2) & c3) <<u c1) */
		if(topbit(c3) >= (32-c2))
			break;
	case 001:	/* (((e >>u c2) & c3) <<u c1) */
		if(c1 > c2) {
			c3 <<= c2;
			c1 -= c2;
			o = OASHL;
			goto rewrite1;
		}
		c3 <<= c1;
		if(c1 == c2)
			goto rewrite0;
		c1 = c2-c1;
		o = OLSHR;
		goto rewrite2;

	case 022:	/* (((e >>s c2) & c3) >>s c1) */
		if(c2 <= 0)
			break;
	case 012:	/* (((e >>s c2) & c3) >>u c1) */
		if(topbit(c3) >= (32-c2))
			break;
		goto s11;
	case 021:	/* (((e >>u c2) & c3) >>s c1) */
		if(topbit(c3) >= 31 && c2 <= 0)
			break;
		goto s11;
	case 011:	/* (((e >>u c2) & c3) >>u c1) */
	s11:
		c3 <<= c2;
		c1 += c2;
		if(c1 >= 32)
			break;
		o = OLSHR;
		goto rewrite1;

	case 020:	/* (((e <<u c2) & c3) >>s c1) */
		if(topbit(c3) >= 31)
			break;
	case 010:	/* (((e <<u c2) & c3) >>u c1) */
		c3 >>= c1;
		if(c1 == c2)
			goto rewrite0;
		if(c1 > c2) {
			c1 -= c2;
			goto rewrite2;
		}
		c1 = c2 - c1;
		o = OASHL;
		goto rewrite2;
	}
	return;

rewrite0:	/* get rid of both shifts */
if(debug['<'])prtree(n, "rewrite0");
	*n = *n->left;
	n->left = n->left->left;
	n->right->vconst = c3;
	return;
rewrite1:	/* get rid of lower shift */
if(debug['<'])prtree(n, "rewrite1");
	n->left->left = n->left->left->left;
	n->left->right->vconst = c3;
	n->right->vconst = c1;
	n->op = o;
	return;
rewrite2:	/* get rid of upper shift */
if(debug['<'])prtree(n, "rewrite2");
	*n = *n->left;
	n->right->vconst = c3;
	n->left->right->vconst = c1;
	n->left->op = o;
}

int
side(Node *n)
{

loop:
	if(n != Z)
	switch(n->op) {
	case OCAST:
	case ONOT:
	case OADDR:
	case OIND:
		n = n->left;
		goto loop;

	case OCOND:
		if(side(n->left))
			break;
		n = n->right;

	case OEQ:
	case ONE:
	case OLT:
	case OGE:
	case OGT:
	case OLE:
	case OADD:
	case OSUB:
	case OMUL:
	case OLMUL:
	case ODIV:
	case OLDIV:
	case OLSHR:
	case OASHL:
	case OASHR:
	case OAND:
	case OOR:
	case OXOR:
	case OMOD:
	case OLMOD:
	case OANDAND:
	case OOROR:
	case OCOMMA:
	case ODOT:
		if(side(n->left))
			break;
		n = n->right;
		goto loop;

	case OSIGN:
	case OSIZE:
	case OCONST:
	case OSTRING:
	case OLSTRING:
	case ONAME:
		return 0;
	}
	return 1;
}

int
vconst(Node *n)
{
	int i;

	if(n == Z)
		goto no;
	if(n->op != OCONST)
		goto no;
	if(n->type == T)
		goto no;
	switch(n->type->etype)
	{
	case TFLOAT:
	case TDOUBLE:
		i = 100;
		if(n->fconst > i || n->fconst < -i)
			goto no;
		i = n->fconst;
		if(i != n->fconst)
			goto no;
		return i;

	case TVLONG:
	case TUVLONG:
		i = n->vconst;
		if(i != n->vconst)
			goto no;
		return i;

	case TCHAR:
	case TUCHAR:
	case TSHORT:
	case TUSHORT:
	case TINT:
	case TUINT:
	case TLONG:
	case TULONG:
	case TIND:
		i = n->vconst;
		if(i != n->vconst)
			goto no;
		return i;
	}
no:
	return -159;	/* first uninteresting constant */
}

/*
 * return log(n) if n is a power of 2 constant
 */
int
xlog2(uvlong v)
{
	int s, i;
	uvlong m;

	s = 0;
	m = MASK(8*sizeof(uvlong));
	for(i=32; i; i>>=1) {
		m >>= i;
		if(!(v & m)) {
			v >>= i;
			s += i;
		}
	}
	if(v == 1)
		return s;
	return -1;
}

int
vlog(Node *n)
{
	if(n->op != OCONST)
		goto bad;
	if(typefd[n->type->etype])
		goto bad;

	return xlog2(n->vconst);

bad:
	return -1;
}

int
topbit(uint32 v)
{
	int i;

	for(i = -1; v; i++)
		v >>= 1;
	return i;
}

/*
 * try to cast a constant down
 * rather than cast a variable up
 * example:
 *	if(c == 'a')
 */
void
relcon(Node *l, Node *r)
{
	vlong v;

	if(l->op != OCONST)
		return;
	if(r->op != OCAST)
		return;
	if(!nilcast(r->left->type, r->type))
		return;
	switch(r->type->etype) {
	default:
		return;
	case TCHAR:
	case TUCHAR:
	case TSHORT:
	case TUSHORT:
		v = convvtox(l->vconst, r->type->etype);
		if(v != l->vconst)
			return;
		break;
	}
	l->type = r->left->type;
	*r = *r->left;
}

int
relindex(int o)
{

	switch(o) {
	default:
		diag(Z, "bad in relindex: %O", o);
	case OEQ: return 0;
	case ONE: return 1;
	case OLE: return 2;
	case OLS: return 3;
	case OLT: return 4;
	case OLO: return 5;
	case OGE: return 6;
	case OHS: return 7;
	case OGT: return 8;
	case OHI: return 9;
	}
}

Node*
invert(Node *n)
{
	Node *i;

	if(n == Z || n->op != OLIST)
		return n;
	i = n;
	for(n = n->left; n != Z; n = n->left) {
		if(n->op != OLIST)
			break;
		i->left = n->right;
		n->right = i;
		i = n;
	}
	i->left = n;
	return i;
}

int
bitno(int32 b)
{
	int i;

	for(i=0; i<32; i++)
		if(b & (1L<<i))
			return i;
	diag(Z, "bad in bitno");
	return 0;
}

int32
typebitor(int32 a, int32 b)
{
	int32 c;

	c = a | b;
	if(a & b)
		if((a & b) == BLONG)
			c |= BVLONG;		/* long long => vlong */
		else
			warn(Z, "once is enough: %Q", a & b);
	return c;
}

void
diag(Node *n, char *fmt, ...)
{
	char buf[STRINGSZ];
	va_list arg;

	va_start(arg, fmt);
	vseprint(buf, buf+sizeof(buf), fmt, arg);
	va_end(arg);
	Bprint(&diagbuf, "%L %s\n", (n==Z)? nearln: n->lineno, buf);

	if(debug['X']){
		Bflush(&diagbuf);
		abort();
	}
	if(n != Z)
	if(debug['v'])
		prtree(n, "diagnostic");

	nerrors++;
	if(nerrors > 10) {
		Bprint(&diagbuf, "too many errors\n");
		errorexit();
	}
}

void
warn(Node *n, char *fmt, ...)
{
	char buf[STRINGSZ];
	va_list arg;

	if(debug['w']) {
		Bprint(&diagbuf, "warning: ");
		va_start(arg, fmt);
		vseprint(buf, buf+sizeof(buf), fmt, arg);
		va_end(arg);
		Bprint(&diagbuf, "%L %s\n", (n==Z)? nearln: n->lineno, buf);

		if(n != Z)
		if(debug['v'])
			prtree(n, "warning");
	}
}

void
yyerror(char *fmt, ...)
{
	char buf[STRINGSZ];
	va_list arg;

	/*
	 * hack to intercept message from yaccpar
	 */
	if(strcmp(fmt, "syntax error") == 0) {
		yyerror("syntax error, last name: %s", symb);
		return;
	}
	va_start(arg, fmt);
	vseprint(buf, buf+sizeof(buf), fmt, arg);
	va_end(arg);
	Bprint(&diagbuf, "%L %s\n", lineno, buf);
	nerrors++;
	if(nerrors > 10) {
		Bprint(&diagbuf, "too many errors\n");
		errorexit();
	}
}

void
fatal(Node *n, char *fmt, ...)
{
	char buf[STRINGSZ];
	va_list arg;

	va_start(arg, fmt);
	vseprint(buf, buf+sizeof(buf), fmt, arg);
	va_end(arg);
	Bprint(&diagbuf, "%L %s\n", (n==Z)? nearln: n->lineno, buf);

	if(debug['X']){
		Bflush(&diagbuf);
		abort();
	}
	if(n != Z)
	if(debug['v'])
		prtree(n, "diagnostic");

	nerrors++;
	errorexit();
}

uint32	thash1	= 0x2edab8c9;
uint32	thash2	= 0x1dc74fb8;
uint32	thash3	= 0x1f241331;
uint32	thash[NALLTYPES];
Init	thashinit[] =
{
	TXXX,		0x17527bbd,	0,
	TCHAR,		0x5cedd32b,	0,
	TUCHAR,		0x552c4454,	0,
	TSHORT,		0x63040b4b,	0,
	TUSHORT,	0x32a45878,	0,
	TINT,		0x4151d5bd,	0,
	TUINT,		0x5ae707d6,	0,
	TLONG,		0x5ef20f47,	0,
	TULONG,		0x36d8eb8f,	0,
	TVLONG,		0x6e5e9590,	0,
	TUVLONG,	0x75910105,	0,
	TFLOAT,		0x25fd7af1,	0,
	TDOUBLE,	0x7c40a1b2,	0,
	TIND,		0x1b832357,	0,
	TFUNC,		0x6babc9cb,	0,
	TARRAY,		0x7c50986d,	0,
	TVOID,		0x44112eff,	0,
	TSTRUCT,	0x7c2da3bf,	0,
	TUNION,		0x3eb25e98,	0,
	TENUM,		0x44b54f61,	0,
	TFILE,		0x19242ac3,	0,
	TOLD,		0x22b15988,	0,
	TDOT,		0x0204f6b3,	0,
	-1,		0,		0,
};

char*	bnames[NALIGN];
Init	bnamesinit[] =
{
	Axxx,	0,	"Axxx",
	Ael1,	0,	"el1",
	Ael2,	0,	"el2",
	Asu2,	0,	"su2",
	Aarg0,	0,	"arg0",
	Aarg1,	0,	"arg1",
	Aarg2,	0,	"arg2",
	Aaut3,	0,	"aut3",
	-1,	0,	0,
};

char*	tnames[NALLTYPES];
Init	tnamesinit[] =
{
	TXXX,		0,	"TXXX",
	TCHAR,		0,	"CHAR",
	TUCHAR,		0,	"UCHAR",
	TSHORT,		0,	"SHORT",
	TUSHORT,	0,	"USHORT",
	TINT,		0,	"INT",
	TUINT,		0,	"UINT",
	TLONG,		0,	"LONG",
	TULONG,		0,	"ULONG",
	TVLONG,		0,	"VLONG",
	TUVLONG,	0,	"UVLONG",
	TFLOAT,		0,	"FLOAT",
	TDOUBLE,	0,	"DOUBLE",
	TIND,		0,	"IND",
	TFUNC,		0,	"FUNC",
	TARRAY,		0,	"ARRAY",
	TVOID,		0,	"VOID",
	TSTRUCT,	0,	"STRUCT",
	TUNION,		0,	"UNION",
	TENUM,		0,	"ENUM",
	TFILE,		0,	"FILE",
	TOLD,		0,	"OLD",
	TDOT,		0,	"DOT",
	-1,		0,	0,
};

char*	gnames[NGTYPES];
Init	gnamesinit[] =
{
	GXXX,			0,	"GXXX",
	GCONSTNT,		0,	"CONST",
	GVOLATILE,		0,	"VOLATILE",
	GVOLATILE|GCONSTNT,	0,	"CONST-VOLATILE",
	-1,			0,	0,
};

char*	qnames[NALLTYPES];
Init	qnamesinit[] =
{
	TXXX,		0,	"TXXX",
	TCHAR,		0,	"CHAR",
	TUCHAR,		0,	"UCHAR",
	TSHORT,		0,	"SHORT",
	TUSHORT,	0,	"USHORT",
	TINT,		0,	"INT",
	TUINT,		0,	"UINT",
	TLONG,		0,	"LONG",
	TULONG,		0,	"ULONG",
	TVLONG,		0,	"VLONG",
	TUVLONG,	0,	"UVLONG",
	TFLOAT,		0,	"FLOAT",
	TDOUBLE,	0,	"DOUBLE",
	TIND,		0,	"IND",
	TFUNC,		0,	"FUNC",
	TARRAY,		0,	"ARRAY",
	TVOID,		0,	"VOID",
	TSTRUCT,	0,	"STRUCT",
	TUNION,		0,	"UNION",
	TENUM,		0,	"ENUM",

	TAUTO,		0,	"AUTO",
	TEXTERN,	0,	"EXTERN",
	TSTATIC,	0,	"STATIC",
	TTYPEDEF,	0,	"TYPEDEF",
	TTYPESTR,	0,	"TYPESTR",
	TREGISTER,	0,	"REGISTER",
	TCONSTNT,	0,	"CONSTNT",
	TVOLATILE,	0,	"VOLATILE",
	TUNSIGNED,	0,	"UNSIGNED",
	TSIGNED,	0,	"SIGNED",
	TDOT,		0,	"DOT",
	TFILE,		0,	"FILE",
	TOLD,		0,	"OLD",
	-1,		0,	0,
};
char*	cnames[NCTYPES];
Init	cnamesinit[] =
{
	CXXX,		0,	"CXXX",
	CAUTO,		0,	"AUTO",
	CEXTERN,	0,	"EXTERN",
	CGLOBL,		0,	"GLOBL",
	CSTATIC,	0,	"STATIC",
	CLOCAL,		0,	"LOCAL",
	CTYPEDEF,	0,	"TYPEDEF",
	CTYPESTR,	0,	"TYPESTR",
	CPARAM,		0,	"PARAM",
	CSELEM,		0,	"SELEM",
	CLABEL,		0,	"LABEL",
	CEXREG,		0,	"EXREG",
	-1,		0,	0,
};

char*	onames[OEND+1];
Init	onamesinit[] =
{
	OXXX,		0,	"OXXX",
	OADD,		0,	"ADD",
	OADDR,		0,	"ADDR",
	OAND,		0,	"AND",
	OANDAND,	0,	"ANDAND",
	OARRAY,		0,	"ARRAY",
	OAS,		0,	"AS",
	OASI,		0,	"ASI",
	OASADD,		0,	"ASADD",
	OASAND,		0,	"ASAND",
	OASASHL,	0,	"ASASHL",
	OASASHR,	0,	"ASASHR",
	OASDIV,		0,	"ASDIV",
	OASHL,		0,	"ASHL",
	OASHR,		0,	"ASHR",
	OASLDIV,	0,	"ASLDIV",
	OASLMOD,	0,	"ASLMOD",
	OASLMUL,	0,	"ASLMUL",
	OASLSHR,	0,	"ASLSHR",
	OASMOD,		0,	"ASMOD",
	OASMUL,		0,	"ASMUL",
	OASOR,		0,	"ASOR",
	OASSUB,		0,	"ASSUB",
	OASXOR,		0,	"ASXOR",
	OBIT,		0,	"BIT",
	OBREAK,		0,	"BREAK",
	OCASE,		0,	"CASE",
	OCAST,		0,	"CAST",
	OCOMMA,		0,	"COMMA",
	OCOND,		0,	"COND",
	OCONST,		0,	"CONST",
	OCONTINUE,	0,	"CONTINUE",
	ODIV,		0,	"DIV",
	ODOT,		0,	"DOT",
	ODOTDOT,	0,	"DOTDOT",
	ODWHILE,	0,	"DWHILE",
	OENUM,		0,	"ENUM",
	OEQ,		0,	"EQ",
	OEXREG,	0,	"EXREG",
	OFOR,		0,	"FOR",
	OFUNC,		0,	"FUNC",
	OGE,		0,	"GE",
	OGOTO,		0,	"GOTO",
	OGT,		0,	"GT",
	OHI,		0,	"HI",
	OHS,		0,	"HS",
	OIF,		0,	"IF",
	OIND,		0,	"IND",
	OINDREG,	0,	"INDREG",
	OINIT,		0,	"INIT",
	OLABEL,		0,	"LABEL",
	OLDIV,		0,	"LDIV",
	OLE,		0,	"LE",
	OLIST,		0,	"LIST",
	OLMOD,		0,	"LMOD",
	OLMUL,		0,	"LMUL",
	OLO,		0,	"LO",
	OLS,		0,	"LS",
	OLSHR,		0,	"LSHR",
	OLT,		0,	"LT",
	OMOD,		0,	"MOD",
	OMUL,		0,	"MUL",
	ONAME,		0,	"NAME",
	ONE,		0,	"NE",
	ONOT,		0,	"NOT",
	OOR,		0,	"OR",
	OOROR,		0,	"OROR",
	OPOSTDEC,	0,	"POSTDEC",
	OPOSTINC,	0,	"POSTINC",
	OPREDEC,	0,	"PREDEC",
	OPREINC,	0,	"PREINC",
	OPREFETCH,		0,	"PREFETCH",
	OPROTO,		0,	"PROTO",
	OREGISTER,	0,	"REGISTER",
	ORETURN,	0,	"RETURN",
	OSET,		0,	"SET",
	OSIGN,		0,	"SIGN",
	OSIZE,		0,	"SIZE",
	OSTRING,	0,	"STRING",
	OLSTRING,	0,	"LSTRING",
	OSTRUCT,	0,	"STRUCT",
	OSUB,		0,	"SUB",
	OSWITCH,	0,	"SWITCH",
	OUNION,		0,	"UNION",
	OUSED,		0,	"USED",
	OWHILE,		0,	"WHILE",
	OXOR,		0,	"XOR",
	OPOS,		0,	"POS",
	ONEG,		0,	"NEG",
	OCOM,		0,	"COM",
	OELEM,		0,	"ELEM",
	OTST,		0,	"TST",
	OINDEX,		0,	"INDEX",
	OFAS,		0,	"FAS",
	OREGPAIR,	0,	"REGPAIR",
	OROTL,		0,	"ROTL",
	OEND,		0,	"END",
	-1,		0,	0,
};

/*	OEQ, ONE, OLE, OLS, OLT, OLO, OGE, OHS, OGT, OHI */
uchar	comrel[12] =
{
	ONE, OEQ, OGT, OHI, OGE, OHS, OLT, OLO, OLE, OLS,
};
uchar	invrel[12] =
{
	OEQ, ONE, OGE, OHS, OGT, OHI, OLE, OLS, OLT, OLO,
};
uchar	logrel[12] =
{
	OEQ, ONE, OLS, OLS, OLO, OLO, OHS, OHS, OHI, OHI,
};

uchar	typei[NALLTYPES];
int	typeiinit[] =
{
	TCHAR, TUCHAR, TSHORT, TUSHORT, TINT, TUINT, TLONG, TULONG, TVLONG, TUVLONG, -1,
};
uchar	typeu[NALLTYPES];
int	typeuinit[] =
{
	TUCHAR, TUSHORT, TUINT, TULONG, TUVLONG, TIND, -1,
};

uchar	typesuv[NALLTYPES];
int	typesuvinit[] =
{
	TVLONG, TUVLONG, TSTRUCT, TUNION, -1,
};

uchar	typeilp[NALLTYPES];
int	typeilpinit[] =
{
	TINT, TUINT, TLONG, TULONG, TIND, -1
};

uchar	typechl[NALLTYPES];
uchar	typechlv[NALLTYPES];
uchar	typechlvp[NALLTYPES];
int	typechlinit[] =
{
	TCHAR, TUCHAR, TSHORT, TUSHORT, TINT, TUINT, TLONG, TULONG, -1,
};

uchar	typechlp[NALLTYPES];
int	typechlpinit[] =
{
	TCHAR, TUCHAR, TSHORT, TUSHORT, TINT, TUINT, TLONG, TULONG, TIND, -1,
};

uchar	typechlpfd[NALLTYPES];
int	typechlpfdinit[] =
{
	TCHAR, TUCHAR, TSHORT, TUSHORT, TINT, TUINT, TLONG, TULONG, TFLOAT, TDOUBLE, TIND, -1,
};

uchar	typec[NALLTYPES];
int	typecinit[] =
{
	TCHAR, TUCHAR, -1
};

uchar	typeh[NALLTYPES];
int	typehinit[] =
{
	TSHORT, TUSHORT, -1,
};

uchar	typeil[NALLTYPES];
int	typeilinit[] =
{
	TINT, TUINT, TLONG, TULONG, -1,
};

uchar	typev[NALLTYPES];
int	typevinit[] =
{
	TVLONG,	TUVLONG, -1,
};

uchar	typefd[NALLTYPES];
int	typefdinit[] =
{
	TFLOAT, TDOUBLE, -1,
};

uchar	typeaf[NALLTYPES];
int	typeafinit[] =
{
	TFUNC, TARRAY, -1,
};

uchar	typesu[NALLTYPES];
int	typesuinit[] =
{
	TSTRUCT, TUNION, -1,
};

int32	tasign[NALLTYPES];
Init	tasigninit[] =
{
	TCHAR,		BNUMBER,	0,
	TUCHAR,		BNUMBER,	0,
	TSHORT,		BNUMBER,	0,
	TUSHORT,	BNUMBER,	0,
	TINT,		BNUMBER,	0,
	TUINT,		BNUMBER,	0,
	TLONG,		BNUMBER,	0,
	TULONG,		BNUMBER,	0,
	TVLONG,		BNUMBER,	0,
	TUVLONG,	BNUMBER,	0,
	TFLOAT,		BNUMBER,	0,
	TDOUBLE,	BNUMBER,	0,
	TIND,		BIND,		0,
	TSTRUCT,	BSTRUCT,	0,
	TUNION,		BUNION,		0,
	-1,		0,		0,
};

int32	tasadd[NALLTYPES];
Init	tasaddinit[] =
{
	TCHAR,		BNUMBER,	0,
	TUCHAR,		BNUMBER,	0,
	TSHORT,		BNUMBER,	0,
	TUSHORT,	BNUMBER,	0,
	TINT,		BNUMBER,	0,
	TUINT,		BNUMBER,	0,
	TLONG,		BNUMBER,	0,
	TULONG,		BNUMBER,	0,
	TVLONG,		BNUMBER,	0,
	TUVLONG,	BNUMBER,	0,
	TFLOAT,		BNUMBER,	0,
	TDOUBLE,	BNUMBER,	0,
	TIND,		BINTEGER,	0,
	-1,		0,		0,
};

int32	tcast[NALLTYPES];
Init	tcastinit[] =
{
	TCHAR,		BNUMBER|BIND|BVOID,	0,
	TUCHAR,		BNUMBER|BIND|BVOID,	0,
	TSHORT,		BNUMBER|BIND|BVOID,	0,
	TUSHORT,	BNUMBER|BIND|BVOID,	0,
	TINT,		BNUMBER|BIND|BVOID,	0,
	TUINT,		BNUMBER|BIND|BVOID,	0,
	TLONG,		BNUMBER|BIND|BVOID,	0,
	TULONG,		BNUMBER|BIND|BVOID,	0,
	TVLONG,		BNUMBER|BIND|BVOID,	0,
	TUVLONG,	BNUMBER|BIND|BVOID,	0,
	TFLOAT,		BNUMBER|BVOID,		0,
	TDOUBLE,	BNUMBER|BVOID,		0,
	TIND,		BINTEGER|BIND|BVOID,	0,
	TVOID,		BVOID,			0,
	TSTRUCT,	BSTRUCT|BVOID,		0,
	TUNION,		BUNION|BVOID,		0,
	-1,		0,			0,
};

int32	tadd[NALLTYPES];
Init	taddinit[] =
{
	TCHAR,		BNUMBER|BIND,	0,
	TUCHAR,		BNUMBER|BIND,	0,
	TSHORT,		BNUMBER|BIND,	0,
	TUSHORT,	BNUMBER|BIND,	0,
	TINT,		BNUMBER|BIND,	0,
	TUINT,		BNUMBER|BIND,	0,
	TLONG,		BNUMBER|BIND,	0,
	TULONG,		BNUMBER|BIND,	0,
	TVLONG,		BNUMBER|BIND,	0,
	TUVLONG,	BNUMBER|BIND,	0,
	TFLOAT,		BNUMBER,	0,
	TDOUBLE,	BNUMBER,	0,
	TIND,		BINTEGER,	0,
	-1,		0,		0,
};

int32	tsub[NALLTYPES];
Init	tsubinit[] =
{
	TCHAR,		BNUMBER,	0,
	TUCHAR,		BNUMBER,	0,
	TSHORT,		BNUMBER,	0,
	TUSHORT,	BNUMBER,	0,
	TINT,		BNUMBER,	0,
	TUINT,		BNUMBER,	0,
	TLONG,		BNUMBER,	0,
	TULONG,		BNUMBER,	0,
	TVLONG,		BNUMBER,	0,
	TUVLONG,	BNUMBER,	0,
	TFLOAT,		BNUMBER,	0,
	TDOUBLE,	BNUMBER,	0,
	TIND,		BINTEGER|BIND,	0,
	-1,		0,		0,
};

int32	tmul[NALLTYPES];
Init	tmulinit[] =
{
	TCHAR,		BNUMBER,	0,
	TUCHAR,		BNUMBER,	0,
	TSHORT,		BNUMBER,	0,
	TUSHORT,	BNUMBER,	0,
	TINT,		BNUMBER,	0,
	TUINT,		BNUMBER,	0,
	TLONG,		BNUMBER,	0,
	TULONG,		BNUMBER,	0,
	TVLONG,		BNUMBER,	0,
	TUVLONG,	BNUMBER,	0,
	TFLOAT,		BNUMBER,	0,
	TDOUBLE,	BNUMBER,	0,
	-1,		0,		0,
};

int32	tand[NALLTYPES];
Init	tandinit[] =
{
	TCHAR,		BINTEGER,	0,
	TUCHAR,		BINTEGER,	0,
	TSHORT,		BINTEGER,	0,
	TUSHORT,	BINTEGER,	0,
	TINT,		BNUMBER,	0,
	TUINT,		BNUMBER,	0,
	TLONG,		BINTEGER,	0,
	TULONG,		BINTEGER,	0,
	TVLONG,		BINTEGER,	0,
	TUVLONG,	BINTEGER,	0,
	-1,		0,		0,
};

int32	trel[NALLTYPES];
Init	trelinit[] =
{
	TCHAR,		BNUMBER,	0,
	TUCHAR,		BNUMBER,	0,
	TSHORT,		BNUMBER,	0,
	TUSHORT,	BNUMBER,	0,
	TINT,		BNUMBER,	0,
	TUINT,		BNUMBER,	0,
	TLONG,		BNUMBER,	0,
	TULONG,		BNUMBER,	0,
	TVLONG,		BNUMBER,	0,
	TUVLONG,	BNUMBER,	0,
	TFLOAT,		BNUMBER,	0,
	TDOUBLE,	BNUMBER,	0,
	TIND,		BIND,		0,
	-1,		0,		0,
};

int32	tfunct[1] =
{
	BFUNC,
};

int32	tindir[1] =
{
	BIND,
};

int32	tdot[1] =
{
	BSTRUCT|BUNION,
};

int32	tnot[1] =
{
	BNUMBER|BIND,
};

int32	targ[1] =
{
	BNUMBER|BIND|BSTRUCT|BUNION,
};

uchar	tab[NTYPE][NTYPE] =
{
/*TXXX*/	{ 0,
		},

/*TCHAR*/	{ 0,	TCHAR, TUCHAR, TSHORT, TUSHORT, TINT, TUINT, TLONG,
			TULONG, TVLONG, TUVLONG, TFLOAT, TDOUBLE, TIND,
		},
/*TUCHAR*/	{ 0,	TUCHAR, TUCHAR, TUSHORT, TUSHORT, TUINT, TUINT, TULONG,
			TULONG, TUVLONG, TUVLONG, TFLOAT, TDOUBLE, TIND,
		},
/*TSHORT*/	{ 0,	TSHORT, TUSHORT, TSHORT, TUSHORT, TINT, TUINT, TLONG,
			TULONG, TVLONG, TUVLONG, TFLOAT, TDOUBLE, TIND,
		},
/*TUSHORT*/	{ 0,	TUSHORT, TUSHORT, TUSHORT, TUSHORT, TUINT, TUINT, TULONG,
			TULONG, TUVLONG, TUVLONG, TFLOAT, TDOUBLE, TIND,
		},
/*TINT*/	{ 0,	TINT, TUINT, TINT, TUINT, TINT, TUINT, TLONG,
			TULONG, TVLONG, TUVLONG, TFLOAT, TDOUBLE, TIND,
		},
/*TUINT*/	{ 0,	TUINT, TUINT, TUINT, TUINT, TUINT, TUINT, TULONG,
			TULONG, TUVLONG, TUVLONG, TFLOAT, TDOUBLE, TIND,
		},
/*TLONG*/	{ 0,	TLONG, TULONG, TLONG, TULONG, TLONG, TULONG, TLONG,
			TULONG, TVLONG, TUVLONG, TFLOAT, TDOUBLE, TIND,
		},
/*TULONG*/	{ 0,	TULONG, TULONG, TULONG, TULONG, TULONG, TULONG, TULONG,
			TULONG, TUVLONG, TUVLONG, TFLOAT, TDOUBLE, TIND,
		},
/*TVLONG*/	{ 0,	TVLONG, TUVLONG, TVLONG, TUVLONG, TVLONG, TUVLONG, TVLONG,
			TUVLONG, TVLONG, TUVLONG, TFLOAT, TDOUBLE, TIND,
		},
/*TUVLONG*/	{ 0,	TUVLONG, TUVLONG, TUVLONG, TUVLONG, TUVLONG, TUVLONG, TUVLONG,
			TUVLONG, TUVLONG, TUVLONG, TFLOAT, TDOUBLE, TIND,
		},
/*TFLOAT*/	{ 0,	TFLOAT, TFLOAT, TFLOAT, TFLOAT, TFLOAT, TFLOAT, TFLOAT,
			TFLOAT, TFLOAT, TFLOAT, TFLOAT, TDOUBLE, TIND,
		},
/*TDOUBLE*/	{ 0,	TDOUBLE, TDOUBLE, TDOUBLE, TDOUBLE, TDOUBLE, TDOUBLE, TDOUBLE,
			TDOUBLE, TDOUBLE, TDOUBLE, TFLOAT, TDOUBLE, TIND,
		},
/*TIND*/	{ 0,	TIND, TIND, TIND, TIND, TIND, TIND, TIND,
			 TIND, TIND, TIND, TIND, TIND, TIND,
		},
};

void
urk(char *name, int max, int i)
{
	if(i >= max) {
		fprint(2, "bad tinit: %s %d>=%d\n", name, i, max);
		exits("init");
	}
}

void
tinit(void)
{
	int *ip;
	Init *p;

	for(p=thashinit; p->code >= 0; p++) {
		urk("thash", nelem(thash), p->code);
		thash[p->code] = p->value;
	}
	for(p=bnamesinit; p->code >= 0; p++) {
		urk("bnames", nelem(bnames), p->code);
		bnames[p->code] = p->s;
	}
	for(p=tnamesinit; p->code >= 0; p++) {
		urk("tnames", nelem(tnames), p->code);
		tnames[p->code] = p->s;
	}
	for(p=gnamesinit; p->code >= 0; p++) {
		urk("gnames", nelem(gnames), p->code);
		gnames[p->code] = p->s;
	}
	for(p=qnamesinit; p->code >= 0; p++) {
		urk("qnames", nelem(qnames), p->code);
		qnames[p->code] = p->s;
	}
	for(p=cnamesinit; p->code >= 0; p++) {
		urk("cnames", nelem(cnames), p->code);
		cnames[p->code] = p->s;
	}
	for(p=onamesinit; p->code >= 0; p++) {
		urk("onames", nelem(onames), p->code);
		onames[p->code] = p->s;
	}
	for(ip=typeiinit; *ip>=0; ip++) {
		urk("typei", nelem(typei), *ip);
		typei[*ip] = 1;
	}
	for(ip=typeuinit; *ip>=0; ip++) {
		urk("typeu", nelem(typeu), *ip);
		typeu[*ip] = 1;
	}
	for(ip=typesuvinit; *ip>=0; ip++) {
		urk("typesuv", nelem(typesuv), *ip);
		typesuv[*ip] = 1;
	}
	for(ip=typeilpinit; *ip>=0; ip++) {
		urk("typeilp", nelem(typeilp), *ip);
		typeilp[*ip] = 1;
	}
	for(ip=typechlinit; *ip>=0; ip++) {
		urk("typechl", nelem(typechl), *ip);
		typechl[*ip] = 1;
		typechlv[*ip] = 1;
		typechlvp[*ip] = 1;
	}
	for(ip=typechlpinit; *ip>=0; ip++) {
		urk("typechlp", nelem(typechlp), *ip);
		typechlp[*ip] = 1;
		typechlvp[*ip] = 1;
	}
	for(ip=typechlpfdinit; *ip>=0; ip++) {
		urk("typechlpfd", nelem(typechlpfd), *ip);
		typechlpfd[*ip] = 1;
	}
	for(ip=typecinit; *ip>=0; ip++) {
		urk("typec", nelem(typec), *ip);
		typec[*ip] = 1;
	}
	for(ip=typehinit; *ip>=0; ip++) {
		urk("typeh", nelem(typeh), *ip);
		typeh[*ip] = 1;
	}
	for(ip=typeilinit; *ip>=0; ip++) {
		urk("typeil", nelem(typeil), *ip);
		typeil[*ip] = 1;
	}
	for(ip=typevinit; *ip>=0; ip++) {
		urk("typev", nelem(typev), *ip);
		typev[*ip] = 1;
		typechlv[*ip] = 1;
		typechlvp[*ip] = 1;
	}
	for(ip=typefdinit; *ip>=0; ip++) {
		urk("typefd", nelem(typefd), *ip);
		typefd[*ip] = 1;
	}
	for(ip=typeafinit; *ip>=0; ip++) {
		urk("typeaf", nelem(typeaf), *ip);
		typeaf[*ip] = 1;
	}
	for(ip=typesuinit; *ip >= 0; ip++) {
		urk("typesu", nelem(typesu), *ip);
		typesu[*ip] = 1;
	}
	for(p=tasigninit; p->code >= 0; p++) {
		urk("tasign", nelem(tasign), p->code);
		tasign[p->code] = p->value;
	}
	for(p=tasaddinit; p->code >= 0; p++) {
		urk("tasadd", nelem(tasadd), p->code);
		tasadd[p->code] = p->value;
	}
	for(p=tcastinit; p->code >= 0; p++) {
		urk("tcast", nelem(tcast), p->code);
		tcast[p->code] = p->value;
	}
	for(p=taddinit; p->code >= 0; p++) {
		urk("tadd", nelem(tadd), p->code);
		tadd[p->code] = p->value;
	}
	for(p=tsubinit; p->code >= 0; p++) {
		urk("tsub", nelem(tsub), p->code);
		tsub[p->code] = p->value;
	}
	for(p=tmulinit; p->code >= 0; p++) {
		urk("tmul", nelem(tmul), p->code);
		tmul[p->code] = p->value;
	}
	for(p=tandinit; p->code >= 0; p++) {
		urk("tand", nelem(tand), p->code);
		tand[p->code] = p->value;
	}
	for(p=trelinit; p->code >= 0; p++) {
		urk("trel", nelem(trel), p->code);
		trel[p->code] = p->value;
	}
	
	/* 32-bit defaults */
	typeword = typechlp;
	typecmplx = typesuv;
}

/*
 * return 1 if it is impossible to jump into the middle of n.
 */
static int
deadhead(Node *n, int caseok)
{
loop:
	if(n == Z)
		return 1;
	switch(n->op) {
	case OLIST:
		if(!deadhead(n->left, caseok))
			return 0;
	rloop:
		n = n->right;
		goto loop;

	case ORETURN:
		break;

	case OLABEL:
		return 0;

	case OGOTO:
		break;

	case OCASE:
		if(!caseok)
			return 0;
		goto rloop;

	case OSWITCH:
		return deadhead(n->right, 1);

	case OWHILE:
	case ODWHILE:
		goto rloop;

	case OFOR:
		goto rloop;

	case OCONTINUE:
		break;

	case OBREAK:
		break;

	case OIF:
		return deadhead(n->right->left, caseok) && deadhead(n->right->right, caseok);

	case OSET:
	case OUSED:
		break;
	}
	return 1;
}

int
deadheads(Node *c)
{
	return deadhead(c->left, 0) && deadhead(c->right, 0);
}

int
mixedasop(Type *l, Type *r)
{
	return !typefd[l->etype] && typefd[r->etype];
}

LSym*
linksym(Sym *s)
{
	if(s == nil)
		return nil;
	if(s->lsym != nil)
		return s->lsym;
	return linklookup(ctxt, s->name, s->class == CSTATIC);
}
