// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
 * static initialization
 */

#include	<u.h>
#include	<libc.h>
#include	"go.h"

enum
{
	InitNotStarted = 0,
	InitDone = 1,
	InitPending = 2,
};

static void initplan(Node*);
static NodeList *initlist;
static void init2(Node*, NodeList**);
static void init2list(NodeList*, NodeList**);
static int staticinit(Node*, NodeList**);
static Node *staticname(Type*, int);

// init1 walks the AST starting at n, and accumulates in out
// the list of definitions needing init code in dependency order.
static void
init1(Node *n, NodeList **out)
{
	NodeList *l;
	Node *nv;

	if(n == N)
		return;
	init1(n->left, out);
	init1(n->right, out);
	for(l=n->list; l; l=l->next)
		init1(l->n, out);

	if(n->left && n->type && n->left->op == OTYPE && n->class == PFUNC) {
		// Methods called as Type.Method(receiver, ...).
		// Definitions for method expressions are stored in type->nname.
		init1(n->type->nname, out);
	}

	if(n->op != ONAME)
		return;
	switch(n->class) {
	case PEXTERN:
	case PFUNC:
		break;
	default:
		if(isblank(n) && n->curfn == N && n->defn != N && n->defn->initorder == InitNotStarted) {
			// blank names initialization is part of init() but not
			// when they are inside a function.
			break;
		}
		return;
	}

	if(n->initorder == InitDone)
		return;
	if(n->initorder == InitPending) {
		// Since mutually recursive sets of functions are allowed,
		// we don't necessarily raise an error if n depends on a node
		// which is already waiting for its dependencies to be visited.
		//
		// initlist contains a cycle of identifiers referring to each other.
		// If this cycle contains a variable, then this variable refers to itself.
		// Conversely, if there exists an initialization cycle involving
		// a variable in the program, the tree walk will reach a cycle
		// involving that variable.
		if(n->class != PFUNC) {
			nv = n;
			goto foundinitloop;
		}
		for(l=initlist; l->n!=n; l=l->next) {
			if(l->n->class != PFUNC) {
				nv = l->n;
				goto foundinitloop;
			}
		}
		// The loop involves only functions, ok.
		return;

	foundinitloop:
		// if there have already been errors printed,
		// those errors probably confused us and
		// there might not be a loop.  let the user
		// fix those first.
		flusherrors();
		if(nerrors > 0)
			errorexit();

		// There is a loop involving nv. We know about
		// n and initlist = n1 <- ... <- nv <- ... <- n <- ...
		print("%L: initialization loop:\n", nv->lineno);
		// Build back pointers in initlist.
		for(l=initlist; l; l=l->next)
			if(l->next != nil)
				l->next->end = l;
		// Print nv -> ... -> n1 -> n.
		for(l=initlist; l->n!=nv; l=l->next);
		for(; l; l=l->end)
			print("\t%L %S refers to\n", l->n->lineno, l->n->sym);
		// Print n -> ... -> nv.
		for(l=initlist; l->n!=n; l=l->next);
		for(; l->n != nv; l=l->end)
			print("\t%L %S refers to\n", l->n->lineno, l->n->sym);
		print("\t%L %S\n", nv->lineno, nv->sym);
		errorexit();
	}

	// reached a new unvisited node.
	n->initorder = InitPending;
	l = malloc(sizeof *l);
	if(l == nil) {
		flusherrors();
		yyerror("out of memory");
		errorexit();
	}
	l->next = initlist;
	l->n = n;
	l->end = nil;
	initlist = l;

	// make sure that everything n depends on is initialized.
	// n->defn is an assignment to n
	if(n->defn != N) {
		switch(n->defn->op) {
		default:
			goto bad;

		case ODCLFUNC:
			init2list(n->defn->nbody, out);
			break;

		case OAS:
			if(n->defn->left != n)
				goto bad;
			if(isblank(n->defn->left) && candiscard(n->defn->right)) {
				n->defn->op = OEMPTY;
				n->defn->left = N;
				n->defn->right = N;
				break;
			}

			init2(n->defn->right, out);
			if(debug['j'])
				print("%S\n", n->sym);
			if(isblank(n) || !staticinit(n, out)) {
				if(debug['%'])
					dump("nonstatic", n->defn);
				*out = list(*out, n->defn);
			}
			break;

		case OAS2FUNC:
		case OAS2MAPR:
		case OAS2DOTTYPE:
		case OAS2RECV:
			if(n->defn->initorder != InitNotStarted)
				break;
			n->defn->initorder = InitDone;
			for(l=n->defn->rlist; l; l=l->next)
				init1(l->n, out);
			if(debug['%']) dump("nonstatic", n->defn);
			*out = list(*out, n->defn);
			break;
		}
	}
	l = initlist;
	initlist = l->next;
	if(l->n != n)
		fatal("bad initlist");
	free(l);
	n->initorder = InitDone;
	return;

bad:
	dump("defn", n->defn);
	fatal("init1: bad defn");
}

// recurse over n, doing init1 everywhere.
static void
init2(Node *n, NodeList **out)
{
	if(n == N || n->initorder == InitDone)
		return;

	if(n->op == ONAME && n->ninit)
		fatal("name %S with ninit: %+N\n", n->sym, n);

	init1(n, out);
	init2(n->left, out);
	init2(n->right, out);
	init2(n->ntest, out);
	init2list(n->ninit, out);
	init2list(n->list, out);
	init2list(n->rlist, out);
	init2list(n->nbody, out);
	init2list(n->nelse, out);
	
	if(n->op == OCLOSURE)
		init2list(n->closure->nbody, out);
	if(n->op == ODOTMETH || n->op == OCALLPART)
		init2(n->type->nname, out);
}

static void
init2list(NodeList *l, NodeList **out)
{
	for(; l; l=l->next)
		init2(l->n, out);
}

static void
initreorder(NodeList *l, NodeList **out)
{
	Node *n;

	for(; l; l=l->next) {
		n = l->n;
		switch(n->op) {
		case ODCLFUNC:
		case ODCLCONST:
		case ODCLTYPE:
			continue;
		}
		initreorder(n->ninit, out);
		n->ninit = nil;
		init1(n, out);
	}
}

// initfix computes initialization order for a list l of top-level
// declarations and outputs the corresponding list of statements
// to include in the init() function body.
NodeList*
initfix(NodeList *l)
{
	NodeList *lout;
	int lno;

	lout = nil;
	lno = lineno;
	initreorder(l, &lout);
	lineno = lno;
	return lout;
}

/*
 * compilation of top-level (static) assignments
 * into DATA statements if at all possible.
 */

static int staticassign(Node*, Node*, NodeList**);

static int
staticinit(Node *n, NodeList **out)
{
	Node *l, *r;

	if(n->op != ONAME || n->class != PEXTERN || n->defn == N || n->defn->op != OAS)
		fatal("staticinit");

	lineno = n->lineno;
	l = n->defn->left;
	r = n->defn->right;
	return staticassign(l, r, out);
}

// like staticassign but we are copying an already
// initialized value r.
static int
staticcopy(Node *l, Node *r, NodeList **out)
{
	int i;
	InitEntry *e;
	InitPlan *p;
	Node *a, *ll, *rr, *orig, n1;

	if(r->op != ONAME || r->class != PEXTERN || r->sym->pkg != localpkg)
		return 0;
	if(r->defn == N)	// probably zeroed but perhaps supplied externally and of unknown value
		return 0;
	if(r->defn->op != OAS)
		return 0;
	orig = r;
	r = r->defn->right;
	
	switch(r->op) {
	case ONAME:
		if(staticcopy(l, r, out))
			return 1;
		*out = list(*out, nod(OAS, l, r));
		return 1;
	
	case OLITERAL:
		if(iszero(r))
			return 1;
		arch.gdata(l, r, l->type->width);
		return 1;

	case OADDR:
		switch(r->left->op) {
		case ONAME:
			arch.gdata(l, r, l->type->width);
			return 1;
		}
		break;
	
	case OPTRLIT:
		switch(r->left->op) {
		default:
			//dump("not static addr", r);
			break;
		case OARRAYLIT:
		case OSTRUCTLIT:
		case OMAPLIT:
			// copy pointer
			arch.gdata(l, nod(OADDR, r->nname, N), l->type->width);
			return 1;
		}
		break;

	case OARRAYLIT:
		if(isslice(r->type)) {
			// copy slice
			a = r->nname;
			n1 = *l;
			n1.xoffset = l->xoffset + Array_array;
			arch.gdata(&n1, nod(OADDR, a, N), widthptr);
			n1.xoffset = l->xoffset + Array_nel;
			arch.gdata(&n1, r->right, widthint);
			n1.xoffset = l->xoffset + Array_cap;
			arch.gdata(&n1, r->right, widthint);
			return 1;
		}
		// fall through
	case OSTRUCTLIT:
		p = r->initplan;
		n1 = *l;
		for(i=0; i<p->len; i++) {
			e = &p->e[i];
			n1.xoffset = l->xoffset + e->xoffset;
			n1.type = e->expr->type;
			if(e->expr->op == OLITERAL)
				arch.gdata(&n1, e->expr, n1.type->width);
			else {
				ll = nod(OXXX, N, N);
				*ll = n1;
				ll->orig = ll; // completely separate copy
				if(!staticassign(ll, e->expr, out)) {
					// Requires computation, but we're
					// copying someone else's computation.
					rr = nod(OXXX, N, N);
					*rr = *orig;
					rr->orig = rr; // completely separate copy
					rr->type = ll->type;
					rr->xoffset += e->xoffset;
					*out = list(*out, nod(OAS, ll, rr));
				}
			}
		}
		return 1;
	}
	return 0;
}

static int
staticassign(Node *l, Node *r, NodeList **out)
{
	Node *a, n1, nam;
	Type *ta;
	InitPlan *p;
	InitEntry *e;
	int i;
	Strlit *sval;
	
	switch(r->op) {
	default:
		//dump("not static", r);
		break;
	
	case ONAME:
		if(r->class == PEXTERN && r->sym->pkg == localpkg)
			return staticcopy(l, r, out);
		break;

	case OLITERAL:
		if(iszero(r))
			return 1;
		arch.gdata(l, r, l->type->width);
		return 1;

	case OADDR:
		if(stataddr(&nam, r->left)) {
			n1 = *r;
			n1.left = &nam;
			arch.gdata(l, &n1, l->type->width);
			return 1;
		}
	
	case OPTRLIT:
		switch(r->left->op) {
		default:
			//dump("not static ptrlit", r);
			break;

		case OARRAYLIT:
		case OMAPLIT:
		case OSTRUCTLIT:
			// Init pointer.
			a = staticname(r->left->type, 1);
			r->nname = a;
			arch.gdata(l, nod(OADDR, a, N), l->type->width);
			// Init underlying literal.
			if(!staticassign(a, r->left, out))
				*out = list(*out, nod(OAS, a, r->left));
			return 1;
		}
		break;

	case OSTRARRAYBYTE:
		if(l->class == PEXTERN && r->left->op == OLITERAL) {
			sval = r->left->val.u.sval;
			slicebytes(l, sval->s, sval->len);
			return 1;
		}
		break;

	case OARRAYLIT:
		initplan(r);
		if(isslice(r->type)) {
			// Init slice.
			ta = typ(TARRAY);
			ta->type = r->type->type;
			ta->bound = mpgetfix(r->right->val.u.xval);
			a = staticname(ta, 1);
			r->nname = a;
			n1 = *l;
			n1.xoffset = l->xoffset + Array_array;
			arch.gdata(&n1, nod(OADDR, a, N), widthptr);
			n1.xoffset = l->xoffset + Array_nel;
			arch.gdata(&n1, r->right, widthint);
			n1.xoffset = l->xoffset + Array_cap;
			arch.gdata(&n1, r->right, widthint);
			// Fall through to init underlying array.
			l = a;
		}
		// fall through
	case OSTRUCTLIT:
		initplan(r);
		p = r->initplan;
		n1 = *l;
		for(i=0; i<p->len; i++) {
			e = &p->e[i];
			n1.xoffset = l->xoffset + e->xoffset;
			n1.type = e->expr->type;
			if(e->expr->op == OLITERAL)
				arch.gdata(&n1, e->expr, n1.type->width);
			else {
				a = nod(OXXX, N, N);
				*a = n1;
				a->orig = a; // completely separate copy
				if(!staticassign(a, e->expr, out))
					*out = list(*out, nod(OAS, a, e->expr));
			}
		}
		return 1;

	case OMAPLIT:
		// TODO: Table-driven map insert.
		break;
	}
	return 0;
}

/*
 * from here down is the walk analysis
 * of composite literals.
 * most of the work is to generate
 * data statements for the constant
 * part of the composite literal.
 */

static	void	structlit(int ctxt, int pass, Node *n, Node *var, NodeList **init);
static	void	arraylit(int ctxt, int pass, Node *n, Node *var, NodeList **init);
static	void	slicelit(int ctxt, Node *n, Node *var, NodeList **init);
static	void	maplit(int ctxt, Node *n, Node *var, NodeList **init);

static Node*
staticname(Type *t, int ctxt)
{
	Node *n;

	snprint(namebuf, sizeof(namebuf), "statictmp_%.4d", statuniqgen);
	statuniqgen++;
	n = newname(lookup(namebuf));
	if(!ctxt)
		n->readonly = 1;
	addvar(n, t, PEXTERN);
	return n;
}

static int
isliteral(Node *n)
{
	if(n->op == OLITERAL)
		if(n->val.ctype != CTNIL)
			return 1;
	return 0;
}

static int
simplename(Node *n)
{
	if(n->op != ONAME)
		goto no;
	if(!n->addable)
		goto no;
	if(n->class & PHEAP)
		goto no;
	if(n->class == PPARAMREF)
		goto no;
	return 1;

no:
	return 0;
}

static void
litas(Node *l, Node *r, NodeList **init)
{
	Node *a;

	a = nod(OAS, l, r);
	typecheck(&a, Etop);
	walkexpr(&a, init);
	*init = list(*init, a);
}

enum
{
	MODEDYNAM	= 1,
	MODECONST	= 2,
};

static int
getdyn(Node *n, int top)
{
	NodeList *nl;
	Node *value;
	int mode;

	mode = 0;
	switch(n->op) {
	default:
		if(isliteral(n))
			return MODECONST;
		return MODEDYNAM;
	case OARRAYLIT:
		if(!top && n->type->bound < 0)
			return MODEDYNAM;
	case OSTRUCTLIT:
		break;
	}

	for(nl=n->list; nl; nl=nl->next) {
		value = nl->n->right;
		mode |= getdyn(value, 0);
		if(mode == (MODEDYNAM|MODECONST))
			break;
	}
	return mode;
}

static void
structlit(int ctxt, int pass, Node *n, Node *var, NodeList **init)
{
	Node *r, *a;
	NodeList *nl;
	Node *index, *value;

	for(nl=n->list; nl; nl=nl->next) {
		r = nl->n;
		if(r->op != OKEY)
			fatal("structlit: rhs not OKEY: %N", r);
		index = r->left;
		value = r->right;

		switch(value->op) {
		case OARRAYLIT:
			if(value->type->bound < 0) {
				if(pass == 1 && ctxt != 0) {
					a = nod(ODOT, var, newname(index->sym));
					slicelit(ctxt, value, a, init);
				} else
				if(pass == 2 && ctxt == 0) {
					a = nod(ODOT, var, newname(index->sym));
					slicelit(ctxt, value, a, init);
				} else
				if(pass == 3)
					break;
				continue;
			}
			a = nod(ODOT, var, newname(index->sym));
			arraylit(ctxt, pass, value, a, init);
			continue;

		case OSTRUCTLIT:
			a = nod(ODOT, var, newname(index->sym));
			structlit(ctxt, pass, value, a, init);
			continue;
		}

		if(isliteral(value)) {
			if(pass == 2)
				continue;
		} else
			if(pass == 1)
				continue;

		// build list of var.field = expr
		a = nod(ODOT, var, newname(index->sym));
		a = nod(OAS, a, value);
		typecheck(&a, Etop);
		if(pass == 1) {
			walkexpr(&a, init);	// add any assignments in r to top
			if(a->op != OAS)
				fatal("structlit: not as");
			a->dodata = 2;
		} else {
			orderstmtinplace(&a);
			walkstmt(&a);
		}
		*init = list(*init, a);
	}
}

static void
arraylit(int ctxt, int pass, Node *n, Node *var, NodeList **init)
{
	Node *r, *a;
	NodeList *l;
	Node *index, *value;

	for(l=n->list; l; l=l->next) {
		r = l->n;
		if(r->op != OKEY)
			fatal("arraylit: rhs not OKEY: %N", r);
		index = r->left;
		value = r->right;

		switch(value->op) {
		case OARRAYLIT:
			if(value->type->bound < 0) {
				if(pass == 1 && ctxt != 0) {
					a = nod(OINDEX, var, index);
					slicelit(ctxt, value, a, init);
				} else
				if(pass == 2 && ctxt == 0) {
					a = nod(OINDEX, var, index);
					slicelit(ctxt, value, a, init);
				} else
				if(pass == 3)
					break;
				continue;
			}
			a = nod(OINDEX, var, index);
			arraylit(ctxt, pass, value, a, init);
			continue;

		case OSTRUCTLIT:
			a = nod(OINDEX, var, index);
			structlit(ctxt, pass, value, a, init);
			continue;
		}

		if(isliteral(index) && isliteral(value)) {
			if(pass == 2)
				continue;
		} else
			if(pass == 1)
				continue;

		// build list of var[index] = value
		a = nod(OINDEX, var, index);
		a = nod(OAS, a, value);
		typecheck(&a, Etop);
		if(pass == 1) {
			walkexpr(&a, init);
			if(a->op != OAS)
				fatal("arraylit: not as");
			a->dodata = 2;
		} else {
			orderstmtinplace(&a);
			walkstmt(&a);
		}
		*init = list(*init, a);
	}
}

static void
slicelit(int ctxt, Node *n, Node *var, NodeList **init)
{
	Node *r, *a;
	NodeList *l;
	Type *t;
	Node *vstat, *vauto;
	Node *index, *value;
	int mode;

	// make an array type
	t = shallow(n->type);
	t->bound = mpgetfix(n->right->val.u.xval);
	t->width = 0;
	t->sym = nil;
	t->haspointers = 0;
	dowidth(t);

	if(ctxt != 0) {
		// put everything into static array
		vstat = staticname(t, ctxt);
		arraylit(ctxt, 1, n, vstat, init);
		arraylit(ctxt, 2, n, vstat, init);

		// copy static to slice
		a = nod(OSLICE, vstat, nod(OKEY, N, N));
		a = nod(OAS, var, a);
		typecheck(&a, Etop);
		a->dodata = 2;
		*init = list(*init, a);
		return;
	}

	// recipe for var = []t{...}
	// 1. make a static array
	//	var vstat [...]t
	// 2. assign (data statements) the constant part
	//	vstat = constpart{}
	// 3. make an auto pointer to array and allocate heap to it
	//	var vauto *[...]t = new([...]t)
	// 4. copy the static array to the auto array
	//	*vauto = vstat
	// 5. assign slice of allocated heap to var
	//	var = [0:]*auto
	// 6. for each dynamic part assign to the slice
	//	var[i] = dynamic part
	//
	// an optimization is done if there is no constant part
	//	3. var vauto *[...]t = new([...]t)
	//	5. var = [0:]*auto
	//	6. var[i] = dynamic part

	// if the literal contains constants,
	// make static initialized array (1),(2)
	vstat = N;
	mode = getdyn(n, 1);
	if(mode & MODECONST) {
		vstat = staticname(t, ctxt);
		arraylit(ctxt, 1, n, vstat, init);
	}

	// make new auto *array (3 declare)
	vauto = temp(ptrto(t));

	// set auto to point at new temp or heap (3 assign)
	if(n->alloc != N) {
		// temp allocated during order.c for dddarg
		n->alloc->type = t;
		if(vstat == N) {
			a = nod(OAS, n->alloc, N);
			typecheck(&a, Etop);
			*init = list(*init, a);  // zero new temp
		}
		a = nod(OADDR, n->alloc, N);
	} else if(n->esc == EscNone) {
		a = temp(t);
		if(vstat == N) {
			a = nod(OAS, temp(t), N);
			typecheck(&a, Etop);
			*init = list(*init, a);  // zero new temp
			a = a->left;
		}
		a = nod(OADDR, a, N);
	} else {
		a = nod(ONEW, N, N);
		a->list = list1(typenod(t));
	}
	a = nod(OAS, vauto, a);
	typecheck(&a, Etop);
	walkexpr(&a, init);
	*init = list(*init, a);

	if(vstat != N) {
		// copy static to heap (4)
		a = nod(OIND, vauto, N);
		a = nod(OAS, a, vstat);
		typecheck(&a, Etop);
		walkexpr(&a, init);
		*init = list(*init, a);
	}

	// make slice out of heap (5)
	a = nod(OAS, var, nod(OSLICE, vauto, nod(OKEY, N, N)));
	typecheck(&a, Etop);
	orderstmtinplace(&a);
	walkstmt(&a);
	*init = list(*init, a);

	// put dynamics into slice (6)
	for(l=n->list; l; l=l->next) {
		r = l->n;
		if(r->op != OKEY)
			fatal("slicelit: rhs not OKEY: %N", r);
		index = r->left;
		value = r->right;
		a = nod(OINDEX, var, index);
		a->bounded = 1;
		// TODO need to check bounds?

		switch(value->op) {
		case OARRAYLIT:
			if(value->type->bound < 0)
				break;
			arraylit(ctxt, 2, value, a, init);
			continue;

		case OSTRUCTLIT:
			structlit(ctxt, 2, value, a, init);
			continue;
		}

		if(isliteral(index) && isliteral(value))
			continue;

		// build list of var[c] = expr
		a = nod(OAS, a, value);
		typecheck(&a, Etop);
		orderstmtinplace(&a);
		walkstmt(&a);
		*init = list(*init, a);
	}
}

static void
maplit(int ctxt, Node *n, Node *var, NodeList **init)
{
	Node *r, *a;
	NodeList *l;
	int nerr;
	int64 b;
	Type *t, *tk, *tv, *t1;
	Node *vstat, *index, *value, *key, *val;
	Sym *syma, *symb;

USED(ctxt);
ctxt = 0;

	// make the map var
	nerr = nerrors;

	a = nod(OMAKE, N, N);
	a->list = list1(typenod(n->type));
	litas(var, a, init);

	// count the initializers
	b = 0;
	for(l=n->list; l; l=l->next) {
		r = l->n;

		if(r->op != OKEY)
			fatal("maplit: rhs not OKEY: %N", r);
		index = r->left;
		value = r->right;

		if(isliteral(index) && isliteral(value))
			b++;
	}

	if(b != 0) {
		// build type [count]struct { a Tindex, b Tvalue }
		t = n->type;
		tk = t->down;
		tv = t->type;

		symb = lookup("b");
		t = typ(TFIELD);
		t->type = tv;
		t->sym = symb;

		syma = lookup("a");
		t1 = t;
		t = typ(TFIELD);
		t->type = tk;
		t->sym = syma;
		t->down = t1;

		t1 = t;
		t = typ(TSTRUCT);
		t->type = t1;

		t1 = t;
		t = typ(TARRAY);
		t->bound = b;
		t->type = t1;

		dowidth(t);

		// make and initialize static array
		vstat = staticname(t, ctxt);
		b = 0;
		for(l=n->list; l; l=l->next) {
			r = l->n;

			if(r->op != OKEY)
				fatal("maplit: rhs not OKEY: %N", r);
			index = r->left;
			value = r->right;

			if(isliteral(index) && isliteral(value)) {
				// build vstat[b].a = key;
				a = nodintconst(b);
				a = nod(OINDEX, vstat, a);
				a = nod(ODOT, a, newname(syma));
				a = nod(OAS, a, index);
				typecheck(&a, Etop);
				walkexpr(&a, init);
				a->dodata = 2;
				*init = list(*init, a);

				// build vstat[b].b = value;
				a = nodintconst(b);
				a = nod(OINDEX, vstat, a);
				a = nod(ODOT, a, newname(symb));
				a = nod(OAS, a, value);
				typecheck(&a, Etop);
				walkexpr(&a, init);
				a->dodata = 2;
				*init = list(*init, a);

				b++;
			}
		}

		// loop adding structure elements to map
		// for i = 0; i < len(vstat); i++ {
		//	map[vstat[i].a] = vstat[i].b
		// }
		index = temp(types[TINT]);

		a = nod(OINDEX, vstat, index);
		a->bounded = 1;
		a = nod(ODOT, a, newname(symb));

		r = nod(OINDEX, vstat, index);
		r->bounded = 1;
		r = nod(ODOT, r, newname(syma));
		r = nod(OINDEX, var, r);

		r = nod(OAS, r, a);

		a = nod(OFOR, N, N);
		a->nbody = list1(r);

		a->ninit = list1(nod(OAS, index, nodintconst(0)));
		a->ntest = nod(OLT, index, nodintconst(t->bound));
		a->nincr = nod(OAS, index, nod(OADD, index, nodintconst(1)));

		typecheck(&a, Etop);
		walkstmt(&a);
		*init = list(*init, a);
	}

	// put in dynamic entries one-at-a-time
	key = nil;
	val = nil;
	for(l=n->list; l; l=l->next) {
		r = l->n;

		if(r->op != OKEY)
			fatal("maplit: rhs not OKEY: %N", r);
		index = r->left;
		value = r->right;

		if(isliteral(index) && isliteral(value))
			continue;
			
		// build list of var[c] = expr.
		// use temporary so that mapassign1 can have addressable key, val.
		if(key == nil) {
			key = temp(var->type->down);
			val = temp(var->type->type);
		}
		a = nod(OAS, key, r->left);
		typecheck(&a, Etop);
		walkstmt(&a);
		*init = list(*init, a);
		a = nod(OAS, val, r->right);
		typecheck(&a, Etop);
		walkstmt(&a);
		*init = list(*init, a);

		a = nod(OAS, nod(OINDEX, var, key), val);
		typecheck(&a, Etop);
		walkstmt(&a);
		*init = list(*init, a);

		if(nerr != nerrors)
			break;
	}
	
	if(key != nil) {
		a = nod(OVARKILL, key, N);
		typecheck(&a, Etop);
		*init = list(*init, a);
		a = nod(OVARKILL, val, N);
		typecheck(&a, Etop);
		*init = list(*init, a);
	}
}

void
anylit(int ctxt, Node *n, Node *var, NodeList **init)
{
	Type *t;
	Node *a, *vstat, *r;

	t = n->type;
	switch(n->op) {
	default:
		fatal("anylit: not lit");

	case OPTRLIT:
		if(!isptr[t->etype])
			fatal("anylit: not ptr");

		if(n->right != N) {
			r = nod(OADDR, n->right, N);
			typecheck(&r, Erv);
		} else {
			r = nod(ONEW, N, N);
			r->typecheck = 1;
			r->type = t;
			r->esc = n->esc;
		}
		walkexpr(&r, init);
		a = nod(OAS, var, r);

		typecheck(&a, Etop);
		*init = list(*init, a);

		var = nod(OIND, var, N);
		typecheck(&var, Erv | Easgn);
		anylit(ctxt, n->left, var, init);
		break;

	case OSTRUCTLIT:
		if(t->etype != TSTRUCT)
			fatal("anylit: not struct");

		if(simplename(var) && count(n->list) > 4) {

			if(ctxt == 0) {
				// lay out static data
				vstat = staticname(t, ctxt);
				structlit(ctxt, 1, n, vstat, init);

				// copy static to var
				a = nod(OAS, var, vstat);
				typecheck(&a, Etop);
				walkexpr(&a, init);
				*init = list(*init, a);

				// add expressions to automatic
				structlit(ctxt, 2, n, var, init);
				break;
			}
			structlit(ctxt, 1, n, var, init);
			structlit(ctxt, 2, n, var, init);
			break;
		}

		// initialize of not completely specified
		if(simplename(var) || count(n->list) < structcount(t)) {
			a = nod(OAS, var, N);
			typecheck(&a, Etop);
			walkexpr(&a, init);
			*init = list(*init, a);
		}
		structlit(ctxt, 3, n, var, init);
		break;

	case OARRAYLIT:
		if(t->etype != TARRAY)
			fatal("anylit: not array");
		if(t->bound < 0) {
			slicelit(ctxt, n, var, init);
			break;
		}

		if(simplename(var) && count(n->list) > 4) {

			if(ctxt == 0) {
				// lay out static data
				vstat = staticname(t, ctxt);
				arraylit(1, 1, n, vstat, init);

				// copy static to automatic
				a = nod(OAS, var, vstat);
				typecheck(&a, Etop);
				walkexpr(&a, init);
				*init = list(*init, a);

				// add expressions to automatic
				arraylit(ctxt, 2, n, var, init);
				break;
			}
			arraylit(ctxt, 1, n, var, init);
			arraylit(ctxt, 2, n, var, init);
			break;
		}

		// initialize of not completely specified
		if(simplename(var) || count(n->list) < t->bound) {
			a = nod(OAS, var, N);
			typecheck(&a, Etop);
			walkexpr(&a, init);
			*init = list(*init, a);
		}
		arraylit(ctxt, 3, n, var, init);
		break;

	case OMAPLIT:
		if(t->etype != TMAP)
			fatal("anylit: not map");
		maplit(ctxt, n, var, init);
		break;
	}
}

int
oaslit(Node *n, NodeList **init)
{
	int ctxt;

	if(n->left == N || n->right == N)
		goto no;
	if(n->left->type == T || n->right->type == T)
		goto no;
	if(!simplename(n->left))
		goto no;
	if(!eqtype(n->left->type, n->right->type))
		goto no;

	// context is init() function.
	// implies generated data executed
	// exactly once and not subject to races.
	ctxt = 0;
//	if(n->dodata == 1)
//		ctxt = 1;

	switch(n->right->op) {
	default:
		goto no;

	case OSTRUCTLIT:
	case OARRAYLIT:
	case OMAPLIT:
		if(vmatch1(n->left, n->right))
			goto no;
		anylit(ctxt, n->right, n->left, init);
		break;
	}
	n->op = OEMPTY;
	return 1;

no:
	// not a special composit literal assignment
	return 0;
}

static int
getlit(Node *lit)
{
	if(smallintconst(lit))
		return mpgetfix(lit->val.u.xval);
	return -1;
}

int
stataddr(Node *nam, Node *n)
{
	int l;

	if(n == N)
		goto no;

	switch(n->op) {

	case ONAME:
		*nam = *n;
		return n->addable;

	case ODOT:
		if(!stataddr(nam, n->left))
			break;
		nam->xoffset += n->xoffset;
		nam->type = n->type;
		return 1;

	case OINDEX:
		if(n->left->type->bound < 0)
			break;
		if(!stataddr(nam, n->left))
			break;
		l = getlit(n->right);
		if(l < 0)
			break;
		// Check for overflow.
		if(n->type->width != 0 && arch.MAXWIDTH/n->type->width <= l)
			break;
 		nam->xoffset += l*n->type->width;
		nam->type = n->type;
		return 1;
	}

no:
	return 0;
}

int
gen_as_init(Node *n)
{
	Node *nr, *nl;
	Node nam, nod1;

	if(n->dodata == 0)
		goto no;

	nr = n->right;
	nl = n->left;
	if(nr == N) {
		if(!stataddr(&nam, nl))
			goto no;
		if(nam.class != PEXTERN)
			goto no;
		goto yes;
	}

	if(nr->type == T || !eqtype(nl->type, nr->type))
		goto no;

	if(!stataddr(&nam, nl))
		goto no;

	if(nam.class != PEXTERN)
		goto no;

	switch(nr->op) {
	default:
		goto no;

	case OCONVNOP:
		nr = nr->left;
		if(nr == N || nr->op != OSLICEARR)
			goto no;
		// fall through
	
	case OSLICEARR:
		if(nr->right->op == OKEY && nr->right->left == N && nr->right->right == N) {
			nr = nr->left;
			goto slice;
		}
		goto no;

	case OLITERAL:
		break;
	}

	switch(nr->type->etype) {
	default:
		goto no;

	case TBOOL:
	case TINT8:
	case TUINT8:
	case TINT16:
	case TUINT16:
	case TINT32:
	case TUINT32:
	case TINT64:
	case TUINT64:
	case TINT:
	case TUINT:
	case TUINTPTR:
	case TPTR32:
	case TPTR64:
	case TFLOAT32:
	case TFLOAT64:
		arch.gdata(&nam, nr, nr->type->width);
		break;

	case TCOMPLEX64:
	case TCOMPLEX128:
		arch.gdatacomplex(&nam, nr->val.u.cval);
		break;

	case TSTRING:
		arch.gdatastring(&nam, nr->val.u.sval);
		break;
	}

yes:
	return 1;

slice:
	arch.gused(N); // in case the data is the dest of a goto
	nl = nr;
	if(nr == N || nr->op != OADDR)
		goto no;
	nr = nr->left;
	if(nr == N || nr->op != ONAME)
		goto no;

	// nr is the array being converted to a slice
	if(nr->type == T || nr->type->etype != TARRAY || nr->type->bound < 0)
		goto no;

	nam.xoffset += Array_array;
	arch.gdata(&nam, nl, types[tptr]->width);

	nam.xoffset += Array_nel-Array_array;
	nodconst(&nod1, types[TINT], nr->type->bound);
	arch.gdata(&nam, &nod1, widthint);

	nam.xoffset += Array_cap-Array_nel;
	arch.gdata(&nam, &nod1, widthint);

	goto yes;

no:
	if(n->dodata == 2) {
		dump("\ngen_as_init", n);
		fatal("gen_as_init couldnt make data statement");
	}
	return 0;
}

static int isvaluelit(Node*);
static InitEntry* entry(InitPlan*);
static void addvalue(InitPlan*, vlong, Node*, Node*);

static void
initplan(Node *n)
{
	InitPlan *p;
	Node *a;
	NodeList *l;

	if(n->initplan != nil)
		return;
	p = mal(sizeof *p);
	n->initplan = p;
	switch(n->op) {
	default:
		fatal("initplan");
	case OARRAYLIT:
		for(l=n->list; l; l=l->next) {
			a = l->n;
			if(a->op != OKEY || !smallintconst(a->left))
				fatal("initplan arraylit");
			addvalue(p, n->type->type->width*mpgetfix(a->left->val.u.xval), N, a->right);
		}
		break;
	case OSTRUCTLIT:
		for(l=n->list; l; l=l->next) {
			a = l->n;
			if(a->op != OKEY || a->left->type == T)
				fatal("initplan structlit");
			addvalue(p, a->left->type->width, N, a->right);
		}
		break;
	case OMAPLIT:
		for(l=n->list; l; l=l->next) {
			a = l->n;
			if(a->op != OKEY)
				fatal("initplan maplit");
			addvalue(p, -1, a->left, a->right);
		}
		break;
	}
}

static void
addvalue(InitPlan *p, vlong xoffset, Node *key, Node *n)
{
	int i;
	InitPlan *q;
	InitEntry *e;

	USED(key);

	// special case: zero can be dropped entirely
	if(iszero(n)) {
		p->zero += n->type->width;
		return;
	}
	
	// special case: inline struct and array (not slice) literals
	if(isvaluelit(n)) {
		initplan(n);
		q = n->initplan;
		for(i=0; i<q->len; i++) {
			e = entry(p);
			*e = q->e[i];
			e->xoffset += xoffset;
		}
		return;
	}
	
	// add to plan
	if(n->op == OLITERAL)
		p->lit += n->type->width;
	else
		p->expr += n->type->width;

	e = entry(p);
	e->xoffset = xoffset;
	e->expr = n;
}

int
iszero(Node *n)
{
	NodeList *l;

	switch(n->op) {
	case OLITERAL:
		switch(n->val.ctype) {
		default:
			dump("unexpected literal", n);
			fatal("iszero");
	
		case CTNIL:
			return 1;
		
		case CTSTR:
			return n->val.u.sval == nil || n->val.u.sval->len == 0;
	
		case CTBOOL:
			return n->val.u.bval == 0;
			
		case CTINT:
		case CTRUNE:
			return mpcmpfixc(n->val.u.xval, 0) == 0;
	
		case CTFLT:
			return mpcmpfltc(n->val.u.fval, 0) == 0;
	
		case CTCPLX:
			return mpcmpfltc(&n->val.u.cval->real, 0) == 0 && mpcmpfltc(&n->val.u.cval->imag, 0) == 0;
		}
		break;
	case OARRAYLIT:
		if(isslice(n->type))
			break;
		// fall through
	case OSTRUCTLIT:
		for(l=n->list; l; l=l->next)
			if(!iszero(l->n->right))
				return 0;
		return 1;
	}
	return 0;
}

static int
isvaluelit(Node *n)
{
	return (n->op == OARRAYLIT && isfixedarray(n->type)) || n->op == OSTRUCTLIT;
}

static InitEntry*
entry(InitPlan *p)
{
	if(p->len >= p->cap) {
		if(p->cap == 0)
			p->cap = 4;
		else
			p->cap *= 2;
		p->e = realloc(p->e, p->cap*sizeof p->e[0]);
		if(p->e == nil)
			fatal("out of memory");
	}
	return &p->e[p->len++];
}
