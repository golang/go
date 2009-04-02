// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include	"go.h"

enum
{
	Snorm		= 0,
	Strue,
	Sfalse,
	Stype,

	Tdefault,	// default case
	Texprconst,	// normal constant case
	Texprvar,	// normal variable case
	Ttypenil,	// case nil
	Ttypeconst,	// type hashes
	Ttypevar,	// interface type

	Ncase	= 4,	// count needed to split
};

typedef	struct	Case	Case;
struct	Case
{
	Node*	node;		// points at case statement
	uint32	hash;		// hash of a type switch
	uint8	type;		// type of case
	uint8	diag;		// suppress multiple diagnostics
	uint16	ordinal;	// position in switch
	Case*	link;		// linked list to link
};
#define	C	((Case*)nil)

void
dumpcase(Case *c0)
{
	Case *c;

	for(c=c0; c!=C; c=c->link) {
		switch(c->type) {
		case Tdefault:
			print("case-default\n");
			print("	ord=%d\n", c->ordinal);
			break;
		case Texprconst:
			print("case-exprconst\n");
			print("	ord=%d\n", c->ordinal);
			break;
		case Texprvar:
			print("case-exprvar\n");
			print("	ord=%d\n", c->ordinal);
			print("	op=%O\n", c->node->left->op);
			break;
		case Ttypenil:
			print("case-typenil\n");
			print("	ord=%d\n", c->ordinal);
			break;
		case Ttypeconst:
			print("case-typeconst\n");
			print("	ord=%d\n", c->ordinal);
			print("	hash=%ux\n", c->hash);
			break;
		case Ttypevar:
			print("case-typevar\n");
			print("	ord=%d\n", c->ordinal);
			break;
		default:
			print("case-???\n");
			print("	ord=%d\n", c->ordinal);
			print("	op=%O\n", c->node->left->op);
			print("	hash=%ux\n", c->hash);
			break;
		}
	}
	print("\n");
}

static int
ordlcmp(Case *c1, Case *c2)
{
	// sort default first
	if(c1->type == Tdefault)
		return -1;
	if(c2->type == Tdefault)
		return +1;

	// sort nil second
	if(c1->type == Ttypenil)
		return -1;
	if(c2->type == Ttypenil)
		return +1;

	// sort by ordinal
	if(c1->ordinal > c2->ordinal)
		return +1;
	if(c1->ordinal < c2->ordinal)
		return -1;
	return 0;
}

static int
exprcmp(Case *c1, Case *c2)
{
	int ct, n;
	Node *n1, *n2;

	// sort non-constants last
	if(c1->type != Texprconst)
		return +1;
	if(c2->type != Texprconst)
		return -1;

	n1 = c1->node->left;
	n2 = c2->node->left;

	ct = n1->val.ctype;
	if(ct != n2->val.ctype)
		fatal("exprcmp");

	// sort by constant value
	n = 0;
	switch(ct) {
	case CTFLT:
		n = mpcmpfltflt(n1->val.u.fval, n2->val.u.fval);
		break;
	case CTINT:
		n = mpcmpfixfix(n1->val.u.xval, n2->val.u.xval);
		break;
	case CTSTR:
		n = cmpslit(n1, n2);
		break;
	}

	return n;
}

static int
typecmp(Case *c1, Case *c2)
{

	// sort non-constants last
	if(c1->type != Ttypeconst)
		return +1;
	if(c2->type != Ttypeconst)
		return -1;

	// sort by hash code
	if(c1->hash > c2->hash)
		return +1;
	if(c1->hash < c2->hash)
		return -1;
	return 0;
}

static Case*
csort(Case *l, int(*f)(Case*, Case*))
{
	Case *l1, *l2, *le;

	if(l == C || l->link == C)
		return l;

	l1 = l;
	l2 = l;
	for(;;) {
		l2 = l2->link;
		if(l2 == C)
			break;
		l2 = l2->link;
		if(l2 == C)
			break;
		l1 = l1->link;
	}

	l2 = l1->link;
	l1->link = C;
	l1 = csort(l, f);
	l2 = csort(l2, f);

	/* set up lead element */
	if((*f)(l1, l2) < 0) {
		l = l1;
		l1 = l1->link;
	} else {
		l = l2;
		l2 = l2->link;
	}
	le = l;

	for(;;) {
		if(l1 == C) {
			while(l2) {
				le->link = l2;
				le = l2;
				l2 = l2->link;
			}
			le->link = C;
			break;
		}
		if(l2 == C) {
			while(l1) {
				le->link = l1;
				le = l1;
				l1 = l1->link;
			}
			break;
		}
		if((*f)(l1, l2) < 0) {
			le->link = l1;
			le = l1;
			l1 = l1->link;
		} else {
			le->link = l2;
			le = l2;
			l2 = l2->link;
		}
	}
	le->link = C;
	return l;
}

/*
 * walktype
 */
Type*
sw0(Node *c, Type *place, int arg)
{
	if(c == N)
		return T;
	switch(c->op) {
	default:
		if(arg == Stype) {
			yyerror("inappropriate case for a type switch");
			return T;
		}
		walktype(c, Erv);
		break;
	case OTYPESW:
		if(arg != Stype)
			yyerror("inappropriate type case");
		break;
	case OAS:
		yyerror("inappropriate assignment in a case statement");
		break;
	}
	return T;
}

/*
 * return the first type
 */
Type*
sw1(Node *c, Type *place, int arg)
{
	if(place == T)
		return c->type;
	return place;
}

/*
 * return a suitable type
 */
Type*
sw2(Node *c, Type *place, int arg)
{
	return types[TINT];	// botch
}

/*
 * check that switch type
 * is compat with all the cases
 */
Type*
sw3(Node *c, Type *place, int arg)
{
	if(place == T)
		return c->type;
	if(c->type == T)
		c->type = place;
	convlit(c, place);
	if(!ascompat(place, c->type))
		badtype(OSWITCH, place, c->type);
	return place;
}

/*
 * over all cases, call paramenter function.
 * four passes of these are used to allocate
 * types to cases and switch
 */
Type*
walkcases(Node *sw, Type*(*call)(Node*, Type*, int arg), int arg)
{
	Iter save;
	Node *n;
	Type *place;
	int32 lno;

	lno = setlineno(sw);
	place = call(sw->ntest, T, arg);

	n = listfirst(&save, &sw->nbody->left);
	if(n == N || n->op == OEMPTY)
		return T;

loop:
	if(n == N) {
		lineno = lno;
		return place;
	}

	if(n->op != OCASE)
		fatal("walkcases: not case %O\n", n->op);

	if(n->left != N) {
		setlineno(n->left);
		place = call(n->left, place, arg);
	}
	n = listnext(&save);
	goto loop;
}

Node*
newlabel()
{
	static int label;

	label++;
	snprint(namebuf, sizeof(namebuf), "%.6d", label);
	return newname(lookup(namebuf));
}

/*
 * build separate list of statements and cases
 * make labels between cases and statements
 * deal with fallthrough, break, unreachable statements
 */
void
casebody(Node *sw)
{
	Iter save;
	Node *os, *oc, *n, *c;
	Node *cas, *stat, *def;
	Node *go, *br;
	int32 lno;

	lno = setlineno(sw);
	n = listfirst(&save, &sw->nbody);
	if(n == N || n->op == OEMPTY) {
		sw->nbody = nod(OLIST, N, N);
		return;
	}

	cas = N;	// cases
	stat = N;	// statements
	def = N;	// defaults
	os = N;		// last statement
	oc = N;		// last case
	br = nod(OBREAK, N, N);

loop:
	if(n == N) {
		if(oc == N && os != N)
			yyerror("first switch statement must be a case");

		stat = list(stat, br);
		cas = list(cas, def);

		sw->nbody = nod(OLIST, rev(cas), rev(stat));
//dump("case", sw->nbody->left);
//dump("stat", sw->nbody->right);
		lineno = lno;
		return;
	}

	lno = setlineno(n);

	if(n->op != OXCASE) {
		stat = list(stat, n);
		os = n;
		goto next;
	}

	n->op = OCASE;
	if(oc == N && os != N)
		yyerror("first switch statement must be a case");

	// botch - shouldnt fall thru declaration
	if(os != N && os->op == OXFALL)
		os->op = OFALL;
	else
		stat = list(stat, br);

	go = nod(OGOTO, newlabel(), N);

	c = n->left;
	if(c == N) {
		if(def != N)
			yyerror("more than one default case");

		// reuse original default case
		n->right = go;
		def = n;
	}

	// expand multi-valued cases
	for(; c!=N; c=c->right) {
		if(c->op != OLIST) {
			// reuse original case
			n->left = c;
			n->right = go;
			cas = list(cas, n);
			break;
		}
		cas = list(cas, nod(OCASE, c->left, go));
	}
	stat = list(stat, nod(OLABEL, go->left, N));
	oc = n;
	os = N;
	goto next;

next:
	n = listnext(&save);
	goto loop;
}

Case*
mkcaselist(Node *sw, int arg)
{
	Iter save;
	Node *n;
	Case *c, *c1;
	int ord;

	c = C;
	ord = 0;

	n = listfirst(&save, &sw->nbody->left);

loop:
	if(n == N)
		goto done;

	c1 = mal(sizeof(*c1));
	c1->link = c;
	c = c1;

	ord++;
	c->ordinal = ord;
	c->node = n;

	if(n->left == N) {
		c->type = Tdefault;
		goto next;
	}

	switch(arg) {
	case Stype:
		c->hash = 0;
		if(n->left->left == N) {
			c->type = Ttypenil;
			goto next;
		}
		if(istype(n->left->left->type, TINTER)) {
			c->type = Ttypevar;
			goto next;
		}

		c->hash = typehash(n->left->left->type, 1, 0);
		c->type = Ttypeconst;
		goto next;

	case Snorm:
	case Strue:
	case Sfalse:
		c->type = Texprvar;
		switch(consttype(n->left)) {
		case CTFLT:
		case CTINT:
		case CTSTR:
			c->type = Texprconst;
		}
		goto next;
	}
next:
	n = listnext(&save);
	goto loop;

done:
	if(c == C)
		return C;

	// sort by value and diagnose duplicate cases
	switch(arg) {
	case Stype:
		c = csort(c, typecmp);
		for(c1=c; c1->link!=C; c1=c1->link) {
			if(typecmp(c1, c1->link) != 0)
				continue;
			setlineno(c1->link->node);
			yyerror("duplicate case in switch");
			print("    previous case at %L\n",
				c1->node->lineno);
		}
		break;
	case Snorm:
	case Strue:
	case Sfalse:
		c = csort(c, exprcmp);
		for(c1=c; c1->link!=C; c1=c1->link) {
			if(exprcmp(c1, c1->link) != 0)
				continue;
			setlineno(c1->link->node);
			yyerror("duplicate case in switch");
			print("    previous case at %L\n",
				c1->node->lineno);
		}
		break;
	}

	// put list back in processing order
	c = csort(c, ordlcmp);
	return c;
}

static	Node*	exprname;

Node*
exprbsw(Case *c0, int ncase, int arg)
{
	Node *cas;
	Node *a, *n;
	Case *c;
	int i, half;

	cas = N;
	if(ncase < Ncase) {
		for(i=0; i<ncase; i++) {
			n = c0->node;

			switch(arg) {
			case Strue:
				a = nod(OIF, N, N);
				a->ntest = n->left;			// if val
				a->nbody = n->right;			// then goto l
				break;

			case Sfalse:
				a = nod(OIF, N, N);
				a->ntest = nod(ONOT, n->left, N);	// if !val
				a->nbody = n->right;			// then goto l
				break;

			default:
				a = nod(OIF, N, N);
				a->ntest = nod(OEQ, exprname, n->left);	// if name == val
				a->nbody = n->right;			// then goto l
				break;
			}

			cas = list(cas, a);
			c0 = c0->link;
		}
		return cas;
	}

	// find the middle and recur
	c = c0;
	half = ncase>>1;
	for(i=1; i<half; i++)
		c = c->link;
	a = nod(OIF, N, N);
	a->ntest = nod(OLE, exprname, c->node->left);
	a->nbody = exprbsw(c0, half, arg);
	a->nelse = exprbsw(c->link, ncase-half, arg);
	return a;
}

/*
 * normal (expression) switch.
 * rebulid case statements into if .. goto
 */
void
exprswitch(Node *sw)
{
	Node *def, *cas;
	Node *a;
	Case *c0, *c, *c1;
	Type *t;
	int arg, ncase;


	arg = Snorm;
	if(isconst(sw->ntest, CTBOOL)) {
		arg = Strue;
		if(sw->ntest->val.u.bval == 0)
			arg = Sfalse;
	}
	walktype(sw->ntest, Erv);

	/*
	 * pass 0,1,2,3
	 * walk the cases as appropriate for switch type
	 */
	walkcases(sw, sw0, arg);
	t = sw->ntest->type;
	if(t == T)
		t = walkcases(sw, sw1, arg);
	if(t == T)
		t = walkcases(sw, sw2, arg);
	if(t == T)
		return;
	walkcases(sw, sw3, arg);
	convlit(sw->ntest, t);


	/*
	 * convert the switch into OIF statements
	 */
	exprname = N;
	cas = N;
	if(arg != Strue && arg != Sfalse) {
		exprname = nod(OXXX, N, N);
		tempname(exprname, sw->ntest->type);
		cas = nod(OAS, exprname, sw->ntest);
	}

	c0 = mkcaselist(sw, arg);
	if(c0 != C && c0->type == Tdefault) {
		def = c0->node->right;
		c0 = c0->link;
	} else {
		def = nod(OBREAK, N, N);
	}

loop:
	if(c0 == C) {
		cas = list(cas, def);
		sw->nbody->left = rev(cas);
		walkstate(sw->nbody);
		return;
	}

	// deal with the variables one-at-a-time
	if(c0->type != Texprconst) {
		a = exprbsw(c0, 1, arg);
		cas = list(cas, a);
		c0 = c0->link;
		goto loop;
	}

	// do binary search on run of constants
	ncase = 1;
	for(c=c0; c->link!=C; c=c->link) {
		if(c->link->type != Texprconst)
			break;
		ncase++;
	}

	// break the chain at the count
	c1 = c->link;
	c->link = C;

	// sort and compile constants
	c0 = csort(c0, exprcmp);
	a = exprbsw(c0, ncase, arg);
	cas = list(cas, a);

	c0 = c1;
	goto loop;

}

static	Node*	hashname;
static	Node*	facename;
static	Node*	boolname;

Node*
typeone(Node *t)
{
	Node *a, *b;

	a = t->left->left;		// var
	a = nod(OLIST, a, boolname);	// var,bool

	b = nod(ODOTTYPE, facename, N);
	b->type = t->left->left->type;	// interface.(type)

	a = nod(OAS, a, b);		// var,bool = interface.(type)

	b = nod(OIF, N, N);
	b->ntest = boolname;
	b->nbody = t->right;		// if bool { goto l }
	return list(a, b);
}

Node*
typebsw(Case *c0, int ncase)
{
	Node *cas;
	Node *a, *n;
	Case *c;
	int i, half;
	Val v;

	cas = N;

	if(ncase < Ncase) {
		for(i=0; i<ncase; i++) {
			n = c0->node;

			switch(c0->type) {

			case Ttypenil:
				v.ctype = CTNIL;
				a = nod(OIF, N, N);
				a->ntest = nod(OEQ, facename, nodlit(v));
				a->nbody = n->right;		// if i==nil { goto l }
				cas = list(cas, a);
				break;

			case Ttypevar:
				a = typeone(n);
				cas = list(cas, a);
				break;

			case Ttypeconst:
				a = nod(OIF, N, N);
				a->ntest = nod(OEQ, hashname, nodintconst(c0->hash));
				a->nbody = rev(typeone(n));
				cas = list(cas, a);
				break;
			}
			c0 = c0->link;
		}
		return cas;
	}

	// find the middle and recur
	c = c0;
	half = ncase>>1;
	for(i=1; i<half; i++)
		c = c->link;
	a = nod(OIF, N, N);
	a->ntest = nod(OLE, hashname, nodintconst(c->hash));
	a->nbody = typebsw(c0, half);
	a->nelse = typebsw(c->link, ncase-half);
	return a;
}

/*
 * convert switch of the form
 *	switch v := i.(type) { case t1: ..; case t2: ..; }
 * into if statements
 */
void
typeswitch(Node *sw)
{
	Node *cas, *def;
	Node *a;
	Case *c, *c0, *c1;
	int ncase;

	walktype(sw->ntest->right, Erv);
	if(!istype(sw->ntest->right->type, TINTER)) {
		yyerror("type switch must be on an interface");
		return;
	}
	walkcases(sw, sw0, Stype);
	cas = N;

	/*
	 * predeclare temporary variables
	 * and the boolean var
	 */
	facename = nod(OXXX, N, N);
	tempname(facename, sw->ntest->right->type);
	a = nod(OAS, facename, sw->ntest->right);
	cas = list(cas, a);

	boolname = nod(OXXX, N, N);
	tempname(boolname, types[TBOOL]);

	hashname = nod(OXXX, N, N);
	tempname(hashname, types[TUINT32]);

	a = syslook("ifacethash", 1);
	argtype(a, sw->ntest->right->type);
	a = nod(OCALL, a, facename);
	a = nod(OAS, hashname, a);
	cas = list(cas, a);

	c0 = mkcaselist(sw, Stype);
	if(c0 != C && c0->type == Tdefault) {
		def = c0->node->right;
		c0 = c0->link;
	} else {
		def = nod(OBREAK, N, N);
	}

loop:
	if(c0 == C) {
		cas = list(cas, def);
		sw->nbody->left = rev(cas);
		walkstate(sw->nbody);
		return;
	}

	// deal with the variables one-at-a-time
	if(c0->type != Ttypeconst) {
		a = typebsw(c0, 1);
		cas = list(cas, a);
		c0 = c0->link;
		goto loop;
	}

	// do binary search on run of constants
	ncase = 1;
	for(c=c0; c->link!=C; c=c->link) {
		if(c->link->type != Ttypeconst)
			break;
		ncase++;
	}

	// break the chain at the count
	c1 = c->link;
	c->link = C;

	// sort and compile constants
	c0 = csort(c0, typecmp);
	a = typebsw(c0, ncase);
	cas = list(cas, a);

	c0 = c1;
	goto loop;
}

void
walkswitch(Node *sw)
{

	/*
	 * reorder the body into (OLIST, cases, statements)
	 * cases have OGOTO into statements.
	 * both have inserted OBREAK statements
	 */
	walkstate(sw->ninit);
	if(sw->ntest == N)
		sw->ntest = nodbool(1);
	casebody(sw);

	if(sw->ntest->op == OTYPESW) {
		typeswitch(sw);
		return;
	}
	exprswitch(sw);
}
