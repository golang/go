// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
 * static initialization
 */

#include	"go.h"

static NodeList *initlist;
static void init2(Node*, NodeList**);
static void init2list(NodeList*, NodeList**);

static void
init1(Node *n, NodeList **out)
{
	NodeList *l;

	if(n == N)
		return;
	init1(n->left, out);
	init1(n->right, out);
	for(l=n->list; l; l=l->next)
		init1(l->n, out);

	if(n->op != ONAME)
		return;
	switch(n->class) {
	case PEXTERN:
	case PFUNC:
		break;
	default:
		if(isblank(n) && n->defn != N && !n->defn->initorder) {
			n->defn->initorder = 1;
			*out = list(*out, n->defn);
		}
		return;
	}

	if(n->initorder == 1)
		return;
	if(n->initorder == 2) {
		if(n->class == PFUNC)
			return;
		
		// if there have already been errors printed,
		// those errors probably confused us and
		// there might not be a loop.  let the user
		// fix those first.
		flusherrors();
		if(nerrors > 0)
			errorexit();

		print("initialization loop:\n");
		for(l=initlist;; l=l->next) {
			if(l->next == nil)
				break;
			l->next->end = l;
		}
		for(; l; l=l->end)
			print("\t%L %S refers to\n", l->n->lineno, l->n->sym);
		print("\t%L %S\n", n->lineno, n->sym);
		errorexit();
	}
	n->initorder = 2;
	l = malloc(sizeof *l);
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
			n->defn->dodata = 1;
			init1(n->defn->right, out);
			if(debug['j'])
				print("%S\n", n->sym);
			*out = list(*out, n->defn);
			break;
		
		case OAS2FUNC:
		case OAS2MAPR:
		case OAS2DOTTYPE:
		case OAS2RECV:
			if(n->defn->initorder)
				break;
			n->defn->initorder = 1;
			for(l=n->defn->rlist; l; l=l->next)
				init1(l->n, out);
			*out = list(*out, n->defn);
			break;
		}
	}
	l = initlist;
	initlist = l->next;
	if(l->n != n)
		fatal("bad initlist");
	free(l);
	n->initorder = 1;
	return;

bad:
	dump("defn", n->defn);
	fatal("init1: bad defn");
}

// recurse over n, doing init1 everywhere.
static void
init2(Node *n, NodeList **out)
{
	if(n == N || n->initorder == 1)
		return;
	init1(n, out);
	init2(n->left, out);
	init2(n->right, out);
	init2(n->ntest, out);
	init2list(n->ninit, out);
	init2list(n->list, out);
	init2list(n->rlist, out);
	init2list(n->nbody, out);
	init2list(n->nelse, out);
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

NodeList*
initfix(NodeList *l)
{
	NodeList *lout;

	lout = nil;
	initreorder(l, &lout);
	return lout;
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
		walkexpr(&a, init);
		if(pass == 1) {
			if(a->op != OAS)
				fatal("structlit: not as");
			a->dodata = 2;
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
		walkexpr(&a, init);	// add any assignments in r to top
		if(pass == 1) {
			if(a->op != OAS)
				fatal("structlit: not as");
			a->dodata = 2;
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
	vauto = nod(OXXX, N, N);
	tempname(vauto, ptrto(t));

	// set auto to point at new heap (3 assign)
	a = nod(ONEW, N, N);
	a->list = list1(typenod(t));
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
	walkexpr(&a, init);
	*init = list(*init, a);

	// put dynamics into slice (6)
	for(l=n->list; l; l=l->next) {
		r = l->n;
		if(r->op != OKEY)
			fatal("slicelit: rhs not OKEY: %N", r);
		index = r->left;
		value = r->right;
		a = nod(OINDEX, var, index);
		a->etype = 1;	// no bounds checking
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
		walkexpr(&a, init);
		*init = list(*init, a);
	}
}

static void
maplit(int ctxt, Node *n, Node *var, NodeList **init)
{
	Node *r, *a;
	NodeList *l;
	int nerr, b;
	Type *t, *tk, *tv, *t1;
	Node *vstat, *index, *value;
	Sym *syma, *symb;

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
			fatal("slicelit: rhs not OKEY: %N", r);
		index = r->left;
		value = r->right;

		if(isliteral(index) && isliteral(value))
			b++;
	}

	t = T;
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
				fatal("slicelit: rhs not OKEY: %N", r);
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
		index = nod(OXXX, N, N);
		tempname(index, types[TINT]);

		a = nod(OINDEX, vstat, index);
		a->etype = 1;	// no bounds checking
		a = nod(ODOT, a, newname(symb));

		r = nod(OINDEX, vstat, index);
		r->etype = 1;	// no bounds checking
		r = nod(ODOT, r, newname(syma));
		r = nod(OINDEX, var, r);

		r = nod(OAS, r, a);

		a = nod(OFOR, N, N);
		a->nbody = list1(r);

		a->ninit = list1(nod(OAS, index, nodintconst(0)));
		a->ntest = nod(OLT, index, nodintconst(t->bound));
		a->nincr = nod(OASOP, index, nodintconst(1));
		a->nincr->etype = OADD;

		typecheck(&a, Etop);
		walkstmt(&a);
		*init = list(*init, a);
	}

	// put in dynamic entries one-at-a-time
	for(l=n->list; l; l=l->next) {
		r = l->n;

		if(r->op != OKEY)
			fatal("slicelit: rhs not OKEY: %N", r);
		index = r->left;
		value = r->right;

		if(isliteral(index) && isliteral(value))
			continue;

		// build list of var[c] = expr
		a = nod(OINDEX, var, r->left);
		a = nod(OAS, a, r->right);
		typecheck(&a, Etop);
		walkexpr(&a, init);
		if(nerr != nerrors)
			break;

		*init = list(*init, a);
	}
}

void
anylit(int ctxt, Node *n, Node *var, NodeList **init)
{
	Type *t;
	Node *a, *vstat;

	t = n->type;
	switch(n->op) {
	default:
		fatal("anylit: not lit");

	case OSTRUCTLIT:
		if(t->etype != TSTRUCT)
			fatal("anylit: not struct");

		if(simplename(var)) {

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
		if(count(n->list) < structcount(t)) {
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

		if(simplename(var)) {

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
		if(count(n->list) < t->bound) {
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
		gused(N); // in case the data is the dest of a goto
		gdata(&nam, nr, nr->type->width);
		break;

	case TCOMPLEX64:
	case TCOMPLEX128:
		gused(N); // in case the data is the dest of a goto
		gdatacomplex(&nam, nr->val.u.cval);
		break;

	case TSTRING:
		gused(N); // in case the data is the dest of a goto
		gdatastring(&nam, nr->val.u.sval);
		break;
	}

yes:
	return 1;

slice:
	gused(N); // in case the data is the dest of a goto
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
	gdata(&nam, nl, types[tptr]->width);

	nam.xoffset += Array_nel-Array_array;
	nodconst(&nod1, types[TINT32], nr->type->bound);
	gdata(&nam, &nod1, types[TINT32]->width);

	nam.xoffset += Array_cap-Array_nel;
	gdata(&nam, &nod1, types[TINT32]->width);

	goto yes;

no:
	if(n->dodata == 2) {
		dump("\ngen_as_init", n);
		fatal("gen_as_init couldnt make data statement");
	}
	return 0;
}

