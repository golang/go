// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
 * static initialization
 */

#include	"go.h"

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
		if(isblank(n))
			*out = list(*out, n->defn);
		return;
	}

	if(n->initorder == 1)
		return;
	if(n->initorder == 2)
		fatal("init loop");

	// make sure that everything n depends on is initialized.
	// n->defn is an assignment to n
	n->initorder = 2;
	if(n->defn != N) {
		switch(n->defn->op) {
		default:
			goto bad;

		case ODCLFUNC:
			for(l=n->defn->nbody; l; l=l->next)
				init1(l->n, out);
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
		}
	}
	n->initorder = 1;
	return;

bad:
	dump("defn", n->defn);
	fatal("init1: bad defn");
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
 * of composit literals.
 * most of the work is to generate
 * data statements for the constant
 * part of the composit literal.
 */

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

static void	arraylit(Node *n, Node *var, int pass, NodeList **init);

static void
structlit(Node *n, Node *var, int pass, NodeList **init)
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
			if(value->type->bound < 0)
				break;
			a = nod(ODOT, var, newname(index->sym));
			arraylit(value, a, pass, init);
			continue;

		case OSTRUCTLIT:
			a = nod(ODOT, var, newname(index->sym));
			structlit(value, a, pass, init);
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
arraylit(Node *n, Node *var, int pass, NodeList **init)
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
			if(value->type->bound < 0)
				break;
			a = nod(OINDEX, var, index);
			arraylit(value, a, pass, init);
			continue;

		case OSTRUCTLIT:
			a = nod(OINDEX, var, index);
			structlit(value, a, pass, init);
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
slicelit(Node *n, Node *var, NodeList **init)
{
	Node *r, *a;
	NodeList *l;
	Type *t;
	Node *vstat, *vheap;
	Node *index, *value;

	// make an array type
	t = shallow(n->type);
	t->bound = mpgetfix(n->right->val.u.xval);
	t->width = 0;
	dowidth(t);

	// make static initialized array
	vstat = staticname(t);
	arraylit(n, vstat, 1, init);

	// make new *array heap
	vheap = nod(OXXX, N, N);
	tempname(vheap, ptrto(t));

	a = nod(ONEW, N, N);
	a->list = list1(typenod(t));
	a = nod(OAS, vheap, a);
	typecheck(&a, Etop);
	walkexpr(&a, init);
	*init = list(*init, a);

	// copy static to heap
	a = nod(OIND, vheap, N);
	a = nod(OAS, a, vstat);
	typecheck(&a, Etop);
	walkexpr(&a, init);
	*init = list(*init, a);

	// make slice out of heap
	a = nod(OAS, var, vheap);
	typecheck(&a, Etop);
	walkexpr(&a, init);
	*init = list(*init, a);

	// put dynamics into slice
	for(l=n->list; l; l=l->next) {
		r = l->n;
		if(r->op != OKEY)
			fatal("slicelit: rhs not OKEY: %N", r);
		index = r->left;
		value = r->right;

		switch(value->op) {
		case OARRAYLIT:
			if(value->type->bound < 0)
				break;
			a = nod(OINDEX, var, index);
			arraylit(value, a, 2, init);
			continue;

		case OSTRUCTLIT:
			a = nod(OINDEX, var, index);
			structlit(value, a, 2, init);
			continue;
		}

		if(isliteral(index) && isliteral(value))
			continue;

		// build list of var[c] = expr
		a = nod(OINDEX, var, index);
		a = nod(OAS, a, value);
		typecheck(&a, Etop);
		walkexpr(&a, init);	// add any assignments in r to top
		*init = list(*init, a);
	}
}

static void
maplit(Node *n, Node *var, NodeList **init)
{
	Node *r, *a;
	NodeList *l;
	int nerr, b;
	Type *t, *tk, *tv, *t1;
	Node *vstat, *index, *value;
	Sym *syma, *symb;

	// make the map var
	nerr = nerrors;

	a = nod(OMAKE, N, N);
	a->list = list1(typenod(n->type));
	a = nod(OAS, var, a);
	typecheck(&a, Etop);
	walkexpr(&a, init);
	*init = list(*init, a);

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
		vstat = staticname(t);
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
		a = nod(ODOT, a, newname(symb));

		r = nod(OINDEX, vstat, index);
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
anylit(Node *n, Node *var, NodeList **init)
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

			// lay out static data
			vstat = staticname(t);
			structlit(n, vstat, 1, init);

			// copy static to automatic
			a = nod(OAS, var, vstat);
			typecheck(&a, Etop);
			walkexpr(&a, init);
			*init = list(*init, a);

			// add expressions to automatic
			structlit(n, var, 2, init);
			break;
		}

		// initialize of not completely specified
		if(count(n->list) < structcount(t)) {
			a = nod(OAS, var, N);
			typecheck(&a, Etop);
			walkexpr(&a, init);
			*init = list(*init, a);
		}
		structlit(n, var, 3, init);
		break;

	case OARRAYLIT:
		if(t->etype != TARRAY)
			fatal("anylit: not array");
		if(t->bound < 0) {
			slicelit(n, var, init);
			break;
		}

		if(simplename(var)) {

			// lay out static data
			vstat = staticname(t);
			arraylit(n, vstat, 1, init);

			// copy static to automatic
			a = nod(OAS, var, vstat);
			typecheck(&a, Etop);
			walkexpr(&a, init);
			*init = list(*init, a);

			// add expressions to automatic
			arraylit(n, var, 2, init);
			break;
		}

		// initialize of not completely specified
		if(count(n->list) < t->bound) {
			a = nod(OAS, var, N);
			typecheck(&a, Etop);
			walkexpr(&a, init);
			*init = list(*init, a);
		}
		arraylit(n, var, 3, init);
		break;

	case OMAPLIT:
		if(t->etype != TMAP)
			fatal("anylit: not map");
		maplit(n, var, init);
		break;
	}
}

int
oaslit(Node *n, NodeList **init)
{
	Type *t;
	Node *vstat, *a;

	if(n->left == N || n->right == N)
		goto no;
	if(n->left->type == T || n->right->type == T)
		goto no;
	if(!simplename(n->left))
		goto no;
	if(!eqtype(n->left->type, n->right->type))
		goto no;
	if(n->dodata == 1)
		goto initctxt;

	switch(n->right->op) {
	default:
		goto no;

	case OSTRUCTLIT:
	case OARRAYLIT:
	case OMAPLIT:
		if(vmatch1(n->left, n->right))
			goto no;
		anylit(n->right, n->left, init);
		break;
	}
	n->op = OEMPTY;
	return 1;

no:
	// not a special composit literal assignment
	return 0;

initctxt:
	// in the initialization context
	// we are trying to put data statements
	// right into the initialized variables
	switch(n->right->op) {
	default:
		goto no;

	case OSTRUCTLIT:
		structlit(n->right, n->left, 1, init);
		structlit(n->right, n->left, 2, init);
		break;

	case OARRAYLIT:
		t = n->right->type;
		if(t == T)
			goto no;
		if(t->bound >= 0) {
			arraylit(n->right, n->left, 1, init);
			arraylit(n->right, n->left, 2, init);
			break;
		}

		// make a static slice
		// make an array type
		t = shallow(t);
		t->bound = mpgetfix(n->right->right->val.u.xval);
		t->width = 0;
		dowidth(t);

		// make static initialized array
		vstat = staticname(t);
		arraylit(n->right, vstat, 1, init);
		arraylit(n->right, vstat, 2, init);

		// copy static to slice
		a = nod(OADDR, vstat, N);
		a = nod(OAS, n->left, a);
		typecheck(&a, Etop);
// turns into a function that is hard to parse
// in ggen where it is turned into DATA statements
//		walkexpr(&a, init);
		a->dodata = 2;
		*init = list(*init, a);
		break;

	case OMAPLIT:
		maplit(n->right, n->left, init);
		break;
	}
	n->op = OEMPTY;
	return 1;
}

int
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

	case OCONVSLICE:
		goto slice;

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
	case TFLOAT:
		gused(N); // in case the data is the dest of a goto
		gdata(&nam, nr, nr->type->width);
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
	nr = n->right->left;
	if(nr == N || nr->op != OADDR)
		goto no;
	nr = nr->left;
	if(nr == N || nr->op != ONAME)
		goto no;

	// nr is the array being converted to a slice
	if(nr->type == T || nr->type->etype != TARRAY || nr->type->bound < 0)
		goto no;

	nam.xoffset += Array_array;
	gdata(&nam, n->right->left, types[tptr]->width);

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

