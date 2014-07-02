// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include	<u.h>
#include	<libc.h>
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
/*c2go Case *C; */

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

	// sort by type (for switches on interface)
	ct = n1->val.ctype;
	if(ct != n2->val.ctype)
		return ct - n2->val.ctype;
	if(!eqtype(n1->type, n2->type)) {
		if(n1->type->vargen > n2->type->vargen)
			return +1;
		else
			return -1;
	}

	// sort by constant value
	n = 0;
	switch(ct) {
	case CTFLT:
		n = mpcmpfltflt(n1->val.u.fval, n2->val.u.fval);
		break;
	case CTINT:
	case CTRUNE:
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

	// sort by ordinal so duplicate error
	// happens on later case.
	if(c1->ordinal > c2->ordinal)
		return +1;
	if(c1->ordinal < c2->ordinal)
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

static Node*
newlabel(void)
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
static void
casebody(Node *sw, Node *typeswvar)
{
	Node *n, *c, *last;
	Node *def;
	NodeList *cas, *stat, *l, *lc;
	Node *go, *br;
	int32 lno, needvar;

	if(sw->list == nil)
		return;

	lno = setlineno(sw);

	cas = nil;	// cases
	stat = nil;	// statements
	def = N;	// defaults
	br = nod(OBREAK, N, N);

	for(l=sw->list; l; l=l->next) {
		n = l->n;
		setlineno(n);
		if(n->op != OXCASE)
			fatal("casebody %O", n->op);
		n->op = OCASE;
		needvar = count(n->list) != 1 || n->list->n->op == OLITERAL;

		go = nod(OGOTO, newlabel(), N);
		if(n->list == nil) {
			if(def != N)
				yyerror("more than one default case");
			// reuse original default case
			n->right = go;
			def = n;
		}

		if(n->list != nil && n->list->next == nil) {
			// one case - reuse OCASE node.
			c = n->list->n;
			n->left = c;
			n->right = go;
			n->list = nil;
			cas = list(cas, n);
		} else {
			// expand multi-valued cases
			for(lc=n->list; lc; lc=lc->next) {
				c = lc->n;
				cas = list(cas, nod(OCASE, c, go));
			}
		}

		stat = list(stat, nod(OLABEL, go->left, N));
		if(typeswvar && needvar && n->nname != N) {
			NodeList *l;

			l = list1(nod(ODCL, n->nname, N));
			l = list(l, nod(OAS, n->nname, typeswvar));
			typechecklist(l, Etop);
			stat = concat(stat, l);
		}
		stat = concat(stat, n->nbody);

		// botch - shouldn't fall thru declaration
		last = stat->end->n;
		if(last->xoffset == n->xoffset && last->op == OXFALL) {
			if(typeswvar) {
				setlineno(last);
				yyerror("cannot fallthrough in type switch");
			}
			if(l->next == nil) {
				setlineno(last);
				yyerror("cannot fallthrough final case in switch");
			}
			last->op = OFALL;
		} else
			stat = list(stat, br);
	}

	stat = list(stat, br);
	if(def)
		cas = list(cas, def);

	sw->list = cas;
	sw->nbody = stat;
	lineno = lno;
}

static Case*
mkcaselist(Node *sw, int arg)
{
	Node *n;
	Case *c, *c1, *c2;
	NodeList *l;
	int ord;

	c = C;
	ord = 0;

	for(l=sw->list; l; l=l->next) {
		n = l->n;
		c1 = mal(sizeof(*c1));
		c1->link = c;
		c = c1;

		ord++;
		if((uint16)ord != ord)
			fatal("too many cases in switch");
		c->ordinal = ord;
		c->node = n;

		if(n->left == N) {
			c->type = Tdefault;
			continue;
		}

		switch(arg) {
		case Stype:
			c->hash = 0;
			if(n->left->op == OLITERAL) {
				c->type = Ttypenil;
				continue;
			}
			if(istype(n->left->type, TINTER)) {
				c->type = Ttypevar;
				continue;
			}

			c->hash = typehash(n->left->type);
			c->type = Ttypeconst;
			continue;

		case Snorm:
		case Strue:
		case Sfalse:
			c->type = Texprvar;
			c->hash = typehash(n->left->type);
			switch(consttype(n->left)) {
			case CTFLT:
			case CTINT:
			case CTRUNE:
			case CTSTR:
				c->type = Texprconst;
			}
			continue;
		}
	}

	if(c == C)
		return C;

	// sort by value and diagnose duplicate cases
	switch(arg) {
	case Stype:
		c = csort(c, typecmp);
		for(c1=c; c1!=C; c1=c1->link) {
			for(c2=c1->link; c2!=C && c2->hash==c1->hash; c2=c2->link) {
				if(c1->type == Ttypenil || c1->type == Tdefault)
					break;
				if(c2->type == Ttypenil || c2->type == Tdefault)
					break;
				if(!eqtype(c1->node->left->type, c2->node->left->type))
					continue;
				yyerrorl(c2->node->lineno, "duplicate case %T in type switch\n\tprevious case at %L", c2->node->left->type, c1->node->lineno);
			}
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
			yyerror("duplicate case %N in switch\n\tprevious case at %L", c1->node->left, c1->node->lineno);
		}
		break;
	}

	// put list back in processing order
	c = csort(c, ordlcmp);
	return c;
}

static	Node*	exprname;

static Node*
exprbsw(Case *c0, int ncase, int arg)
{
	NodeList *cas;
	Node *a, *n;
	Case *c;
	int i, half, lno;

	cas = nil;
	if(ncase < Ncase) {
		for(i=0; i<ncase; i++) {
			n = c0->node;
			lno = setlineno(n);

			if(assignop(n->left->type, exprname->type, nil) == OCONVIFACE ||
			   assignop(exprname->type, n->left->type, nil) == OCONVIFACE)
				goto snorm;

			switch(arg) {
			case Strue:
				a = nod(OIF, N, N);
				a->ntest = n->left;			// if val
				a->nbody = list1(n->right);			// then goto l
				break;

			case Sfalse:
				a = nod(OIF, N, N);
				a->ntest = nod(ONOT, n->left, N);	// if !val
				typecheck(&a->ntest, Erv);
				a->nbody = list1(n->right);			// then goto l
				break;

			default:
			snorm:
				a = nod(OIF, N, N);
				a->ntest = nod(OEQ, exprname, n->left);	// if name == val
				typecheck(&a->ntest, Erv);
				a->nbody = list1(n->right);			// then goto l
				break;
			}

			cas = list(cas, a);
			c0 = c0->link;
			lineno = lno;
		}
		return liststmt(cas);
	}

	// find the middle and recur
	c = c0;
	half = ncase>>1;
	for(i=1; i<half; i++)
		c = c->link;
	a = nod(OIF, N, N);
	a->ntest = nod(OLE, exprname, c->node->left);
	typecheck(&a->ntest, Erv);
	a->nbody = list1(exprbsw(c0, half, arg));
	a->nelse = list1(exprbsw(c->link, ncase-half, arg));
	return a;
}

/*
 * normal (expression) switch.
 * rebulid case statements into if .. goto
 */
static void
exprswitch(Node *sw)
{
	Node *def;
	NodeList *cas;
	Node *a;
	Case *c0, *c, *c1;
	Type *t;
	int arg, ncase;

	casebody(sw, N);

	arg = Snorm;
	if(isconst(sw->ntest, CTBOOL)) {
		arg = Strue;
		if(sw->ntest->val.u.bval == 0)
			arg = Sfalse;
	}
	walkexpr(&sw->ntest, &sw->ninit);
	t = sw->type;
	if(t == T)
		return;

	/*
	 * convert the switch into OIF statements
	 */
	exprname = N;
	cas = nil;
	if(arg != Strue && arg != Sfalse) {
		exprname = temp(sw->ntest->type);
		cas = list1(nod(OAS, exprname, sw->ntest));
		typechecklist(cas, Etop);
	} else {
		exprname = nodbool(arg == Strue);
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
		sw->nbody = concat(cas, sw->nbody);
		sw->list = nil;
		walkstmtlist(sw->nbody);
		return;
	}

	// deal with the variables one-at-a-time
	if(!okforcmp[t->etype] || c0->type != Texprconst) {
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

static Node*
typeone(Node *t)
{
	NodeList *init;
	Node *a, *b, *var;

	var = t->nname;
	init = nil;
	if(var == N) {
		typecheck(&nblank, Erv | Easgn);
		var = nblank;
	} else
		init = list1(nod(ODCL, var, N));

	a = nod(OAS2, N, N);
	a->list = list(list1(var), boolname);	// var,bool =
	b = nod(ODOTTYPE, facename, N);
	b->type = t->left->type;		// interface.(type)
	a->rlist = list1(b);
	typecheck(&a, Etop);
	init = list(init, a);

	b = nod(OIF, N, N);
	b->ntest = boolname;
	b->nbody = list1(t->right);		// if bool { goto l }
	a = liststmt(list(init, b));
	return a;
}

static Node*
typebsw(Case *c0, int ncase)
{
	NodeList *cas;
	Node *a, *n;
	Case *c;
	int i, half;

	cas = nil;

	if(ncase < Ncase) {
		for(i=0; i<ncase; i++) {
			n = c0->node;
			if(c0->type != Ttypeconst)
				fatal("typebsw");
			a = nod(OIF, N, N);
			a->ntest = nod(OEQ, hashname, nodintconst(c0->hash));
			typecheck(&a->ntest, Erv);
			a->nbody = list1(n->right);
			cas = list(cas, a);
			c0 = c0->link;
		}
		return liststmt(cas);
	}

	// find the middle and recur
	c = c0;
	half = ncase>>1;
	for(i=1; i<half; i++)
		c = c->link;
	a = nod(OIF, N, N);
	a->ntest = nod(OLE, hashname, nodintconst(c->hash));
	typecheck(&a->ntest, Erv);
	a->nbody = list1(typebsw(c0, half));
	a->nelse = list1(typebsw(c->link, ncase-half));
	return a;
}

/*
 * convert switch of the form
 *	switch v := i.(type) { case t1: ..; case t2: ..; }
 * into if statements
 */
static void
typeswitch(Node *sw)
{
	Node *def;
	NodeList *cas, *hash;
	Node *a, *n;
	Case *c, *c0, *c1;
	int ncase;
	Type *t;
	Val v;

	if(sw->ntest == nil)
		return;
	if(sw->ntest->right == nil) {
		setlineno(sw);
		yyerror("type switch must have an assignment");
		return;
	}
	walkexpr(&sw->ntest->right, &sw->ninit);
	if(!istype(sw->ntest->right->type, TINTER)) {
		yyerror("type switch must be on an interface");
		return;
	}
	cas = nil;

	/*
	 * predeclare temporary variables
	 * and the boolean var
	 */
	facename = temp(sw->ntest->right->type);
	a = nod(OAS, facename, sw->ntest->right);
	typecheck(&a, Etop);
	cas = list(cas, a);

	casebody(sw, facename);

	boolname = temp(types[TBOOL]);
	typecheck(&boolname, Erv);

	hashname = temp(types[TUINT32]);
	typecheck(&hashname, Erv);

	t = sw->ntest->right->type;
	if(isnilinter(t))
		a = syslook("efacethash", 1);
	else
		a = syslook("ifacethash", 1);
	argtype(a, t);
	a = nod(OCALL, a, N);
	a->list = list1(facename);
	a = nod(OAS, hashname, a);
	typecheck(&a, Etop);
	cas = list(cas, a);

	c0 = mkcaselist(sw, Stype);
	if(c0 != C && c0->type == Tdefault) {
		def = c0->node->right;
		c0 = c0->link;
	} else {
		def = nod(OBREAK, N, N);
	}
	
	/*
	 * insert if statement into each case block
	 */
	for(c=c0; c!=C; c=c->link) {
		n = c->node;
		switch(c->type) {

		case Ttypenil:
			v.ctype = CTNIL;
			a = nod(OIF, N, N);
			a->ntest = nod(OEQ, facename, nodlit(v));
			typecheck(&a->ntest, Erv);
			a->nbody = list1(n->right);		// if i==nil { goto l }
			n->right = a;
			break;
		
		case Ttypevar:
		case Ttypeconst:
			n->right = typeone(n);
			break;
		}
	}

	/*
	 * generate list of if statements, binary search for constant sequences
	 */
	while(c0 != C) {
		if(c0->type != Ttypeconst) {
			n = c0->node;
			cas = list(cas, n->right);
			c0=c0->link;
			continue;
		}
		
		// identify run of constants
		c1 = c = c0;
		while(c->link!=C && c->link->type==Ttypeconst)
			c = c->link;
		c0 = c->link;
		c->link = nil;

		// sort by hash
		c1 = csort(c1, typecmp);
		
		// for debugging: linear search
		if(0) {
			for(c=c1; c!=C; c=c->link) {
				n = c->node;
				cas = list(cas, n->right);
			}
			continue;
		}

		// combine adjacent cases with the same hash
		ncase = 0;
		for(c=c1; c!=C; c=c->link) {
			ncase++;
			hash = list1(c->node->right);
			while(c->link != C && c->link->hash == c->hash) {
				hash = list(hash, c->link->node->right);
				c->link = c->link->link;
			}
			c->node->right = liststmt(hash);
		}
		
		// binary search among cases to narrow by hash
		cas = list(cas, typebsw(c1, ncase));
	}
	if(nerrors == 0) {
		cas = list(cas, def);
		sw->nbody = concat(cas, sw->nbody);
		sw->list = nil;
		walkstmtlist(sw->nbody);
	}
}

void
walkswitch(Node *sw)
{
	/*
	 * reorder the body into (OLIST, cases, statements)
	 * cases have OGOTO into statements.
	 * both have inserted OBREAK statements
	 */
	if(sw->ntest == N) {
		sw->ntest = nodbool(1);
		typecheck(&sw->ntest, Erv);
	}

	if(sw->ntest->op == OTYPESW) {
		typeswitch(sw);
//dump("sw", sw);
		return;
	}
	exprswitch(sw);
	// Discard old AST elements after a walk. They can confuse racewealk.
	sw->ntest = nil;
	sw->list = nil;
}

/*
 * type check switch statement
 */
void
typecheckswitch(Node *n)
{
	int top, lno, ptr;
	char *nilonly;
	Type *t, *badtype, *missing, *have;
	NodeList *l, *ll;
	Node *ncase, *nvar;
	Node *def;

	lno = lineno;
	typechecklist(n->ninit, Etop);
	nilonly = nil;

	if(n->ntest != N && n->ntest->op == OTYPESW) {
		// type switch
		top = Etype;
		typecheck(&n->ntest->right, Erv);
		t = n->ntest->right->type;
		if(t != T && t->etype != TINTER)
			yyerror("cannot type switch on non-interface value %lN", n->ntest->right);
	} else {
		// value switch
		top = Erv;
		if(n->ntest) {
			typecheck(&n->ntest, Erv);
			defaultlit(&n->ntest, T);
			t = n->ntest->type;
		} else
			t = types[TBOOL];
		if(t) {
			if(!okforeq[t->etype])
				yyerror("cannot switch on %lN", n->ntest);
			else if(t->etype == TARRAY && !isfixedarray(t))
				nilonly = "slice";
			else if(t->etype == TARRAY && isfixedarray(t) && algtype1(t, nil) == ANOEQ)
				yyerror("cannot switch on %lN", n->ntest);
			else if(t->etype == TSTRUCT && algtype1(t, &badtype) == ANOEQ)
				yyerror("cannot switch on %lN (struct containing %T cannot be compared)", n->ntest, badtype);
			else if(t->etype == TFUNC)
				nilonly = "func";
			else if(t->etype == TMAP)
				nilonly = "map";
		}
	}
	n->type = t;

	def = N;
	for(l=n->list; l; l=l->next) {
		ncase = l->n;
		setlineno(n);
		if(ncase->list == nil) {
			// default
			if(def != N)
				yyerror("multiple defaults in switch (first at %L)", def->lineno);
			else
				def = ncase;
		} else {
			for(ll=ncase->list; ll; ll=ll->next) {
				setlineno(ll->n);
				typecheck(&ll->n, Erv | Etype);
				if(ll->n->type == T || t == T)
					continue;
				setlineno(ncase);
				switch(top) {
				case Erv:	// expression switch
					defaultlit(&ll->n, t);
					if(ll->n->op == OTYPE)
						yyerror("type %T is not an expression", ll->n->type);
					else if(ll->n->type != T && !assignop(ll->n->type, t, nil) && !assignop(t, ll->n->type, nil)) {
						if(n->ntest)
							yyerror("invalid case %N in switch on %N (mismatched types %T and %T)", ll->n, n->ntest, ll->n->type, t);
						else
							yyerror("invalid case %N in switch (mismatched types %T and bool)", ll->n, ll->n->type);
					} else if(nilonly && !isconst(ll->n, CTNIL)) {
						yyerror("invalid case %N in switch (can only compare %s %N to nil)", ll->n, nilonly, n->ntest);
					}
					break;
				case Etype:	// type switch
					if(ll->n->op == OLITERAL && istype(ll->n->type, TNIL)) {
						;
					} else if(ll->n->op != OTYPE && ll->n->type != T) {  // should this be ||?
						yyerror("%lN is not a type", ll->n);
						// reset to original type
						ll->n = n->ntest->right;
					} else if(ll->n->type->etype != TINTER && t->etype == TINTER && !implements(ll->n->type, t, &missing, &have, &ptr)) {
						if(have && !missing->broke && !have->broke)
							yyerror("impossible type switch case: %lN cannot have dynamic type %T"
								" (wrong type for %S method)\n\thave %S%hT\n\twant %S%hT",
								n->ntest->right, ll->n->type, missing->sym, have->sym, have->type,
								missing->sym, missing->type);
						else if(!missing->broke)
							yyerror("impossible type switch case: %lN cannot have dynamic type %T"
								" (missing %S method)", n->ntest->right, ll->n->type, missing->sym);
					}
					break;
				}
			}
		}
		if(top == Etype && n->type != T) {
			ll = ncase->list;
			nvar = ncase->nname;
			if(nvar != N) {
				if(ll && ll->next == nil && ll->n->type != T && !istype(ll->n->type, TNIL)) {
					// single entry type switch
					nvar->ntype = typenod(ll->n->type);
				} else {
					// multiple entry type switch or default
					nvar->ntype = typenod(n->type);
				}
			}
		}
		typechecklist(ncase->nbody, Etop);
	}

	lineno = lno;
}
