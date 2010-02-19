// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include	"go.h"
#include	"y.tab.h"

static	void	funcargs(Node*);

int
dflag(void)
{
	if(!debug['d'])
		return 0;
	if(debug['y'])
		return 1;
	if(incannedimport)
		return 0;
	return 1;
}

/*
 * declaration stack & operations
 */
static	Sym*	dclstack;

void
dcopy(Sym *a, Sym *b)
{
	a->pkg = b->pkg;
	a->name = b->name;
	a->def = b->def;
	a->block = b->block;
	a->lastlineno = b->lastlineno;
}

Sym*
push(void)
{
	Sym *d;

	d = mal(sizeof(*d));
	d->link = dclstack;
	dclstack = d;
	return d;
}

Sym*
pushdcl(Sym *s)
{
	Sym *d;

	d = push();
	dcopy(d, s);
	if(dflag())
		print("\t%L push %S %p\n", lineno, s, s->def);
	return d;
}

void
popdcl(void)
{
	Sym *d, *s;

//	if(dflag())
//		print("revert\n");

	for(d=dclstack; d!=S; d=d->link) {
		if(d->name == nil)
			break;
		s = pkglookup(d->name, d->pkg);
		dcopy(s, d);
		if(dflag())
			print("\t%L pop %S %p\n", lineno, s, s->def);
	}
	if(d == S)
		fatal("popdcl: no mark");
	dclstack = d->link;
	block = d->block;
}

void
poptodcl(void)
{
	Sym *d, *s;

	for(d=dclstack; d!=S; d=d->link) {
		if(d->name == nil)
			break;
		s = pkglookup(d->name, d->pkg);
		dcopy(s, d);
		if(dflag())
			print("\t%L pop %S\n", lineno, s);
	}
	if(d == S)
		fatal("poptodcl: no mark");
	dclstack = d;
}

void
markdcl(void)
{
	Sym *d;

	d = push();
	d->name = nil;		// used as a mark in fifo
	d->block = block;

	blockgen++;
	block = blockgen;

//	if(dflag())
//		print("markdcl\n");
}

void
dumpdcl(char *st)
{
	Sym *s, *d;
	int i;

	i = 0;
	for(d=dclstack; d!=S; d=d->link) {
		i++;
		print("    %.2d %p", i, d);
		if(d->name == nil) {
			print("\n");
			continue;
		}
		print(" '%s'", d->name);
		s = pkglookup(d->name, d->pkg);
		print(" %lS\n", s);
	}
}

void
testdclstack(void)
{
	Sym *d;

	for(d=dclstack; d!=S; d=d->link) {
		if(d->name == nil) {
			yyerror("mark left on the stack");
			continue;
		}
	}
}

void
redeclare(Sym *s, char *where)
{
	if(s->lastlineno == 0)
		yyerror("%S redeclared %s\n"
			"\tprevious declaration during import",
			s, where);
	else
		yyerror("%S redeclared %s\n"
			"\tprevious declaration at %L",
			s, where, s->lastlineno);
}

/*
 * declare individual names - var, typ, const
 */
void
declare(Node *n, int ctxt)
{
	Sym *s;
	int gen;
	static int typegen, vargen;

	if(isblank(n))
		return;

	n->lineno = parserline();
	s = n->sym;
	gen = 0;
	if(ctxt == PEXTERN) {
		externdcl = list(externdcl, n);
		if(dflag())
			print("\t%L global decl %S %p\n", lineno, s, n);
	} else {
		if(curfn == nil && ctxt == PAUTO)
			fatal("automatic outside function");
		if(curfn != nil)
			curfn->dcl = list(curfn->dcl, n);
		if(n->op == OTYPE)
			gen = ++typegen;
		else if(n->op == ONAME)
			gen = ++vargen;
		pushdcl(s);
	}
	if(ctxt == PAUTO)
		n->xoffset = BADWIDTH;

	if(s->block == block)
		redeclare(s, "in this block");

	s->block = block;
	s->lastlineno = parserline();
	s->def = n;
	n->vargen = gen;
	n->funcdepth = funcdepth;
	n->class = ctxt;

	autoexport(n, ctxt);
}

void
addvar(Node *n, Type *t, int ctxt)
{
	if(n==N || n->sym == S || (n->op != ONAME && n->op != ONONAME) || t == T)
		fatal("addvar: n=%N t=%T nil", n, t);

	n->op = ONAME;
	declare(n, ctxt);
	n->type = t;
}

// TODO: cut use of below in sigtype and then delete
void
addtyp(Type *n, int ctxt)
{
	Node *def;

	if(n==T || n->sym == S)
		fatal("addtyp: n=%T t=%T nil", n);

	def = typenod(n);
	declare(def, ctxt);
	n->vargen = def->vargen;

	typelist = list(typelist, def);
}

/*
 * introduce a type named n
 * but it is an unknown type for now
 */
// TODO(rsc): cut use of this in sigtype and then delete
Type*
dodcltype(Type *n)
{
	addtyp(n, dclcontext);
	n->local = 1;
	autoexport(typenod(n), dclcontext);
	return n;
}

/*
 * now we know what n is: it's t
 */
// TODO(rsc): cut use of this in sigtype and then delete
void
updatetype(Type *n, Type *t)
{
	Sym *s;
	int local, vargen;
	int maplineno, lno, etype;

	if(t == T)
		return;
	s = n->sym;
	if(s == S || s->def == N || s->def->op != OTYPE || s->def->type != n)
		fatal("updatetype %T = %T", n, t);

	etype = n->etype;
	switch(n->etype) {
	case TFORW:
		break;

	default:
		fatal("updatetype %T / %T", n, t);
	}

	// decl was
	//	type n t;
	// copy t, but then zero out state associated with t
	// that is no longer associated with n.
	maplineno = n->maplineno;
	local = n->local;
	vargen = n->vargen;
	*n = *t;
	n->sym = s;
	n->local = local;
	n->siggen = 0;
	n->printed = 0;
	n->method = nil;
	n->vargen = vargen;
	n->nod = N;

	checkwidth(n);

	// double-check use of type as map key
	if(maplineno) {
		lno = lineno;
		lineno = maplineno;
		maptype(n, types[TBOOL]);
		lineno = lno;
	}
}

/*
 * declare variables from grammar
 * new_name_list (type | [type] = expr_list)
 */
NodeList*
variter(NodeList *vl, Node *t, NodeList *el)
{
	int doexpr;
	Node *v, *e, *as2;
	NodeList *init;

	init = nil;
	doexpr = el != nil;
	
	if(count(el) == 1 && count(vl) > 1) {
		e = el->n;
		as2 = nod(OAS2, N, N);
		as2->list = vl;
		as2->rlist = list1(e);
		for(; vl; vl=vl->next) {
			v = vl->n;
			v->op = ONAME;
			declare(v, dclcontext);
			v->ntype = t;
			v->defn = as2;
			if(funcdepth > 0)
				init = list(init, nod(ODCL, v, N));
		}
		return list(init, as2);
	}
	
	for(; vl; vl=vl->next) {
		if(doexpr) {
			if(el == nil) {
				yyerror("missing expr in var dcl");
				break;
			}
			e = el->n;
			el = el->next;
		} else
			e = N;

		v = vl->n;
		v->op = ONAME;
		declare(v, dclcontext);
		v->ntype = t;

		if(e != N || funcdepth > 0 || isblank(v)) {
			if(funcdepth > 0)
				init = list(init, nod(ODCL, v, N));
			e = nod(OAS, v, e);
			init = list(init, e);
			if(e->right != N)
				v->defn = e;
		}
	}
	if(el != nil)
		yyerror("extra expr in var dcl");
	return init;
}

/*
 * declare constants from grammar
 * new_name_list [[type] = expr_list]
 */
NodeList*
constiter(NodeList *vl, Node *t, NodeList *cl)
{
	Node *v, *c;
	NodeList *vv;

	vv = nil;
	if(cl == nil) {
		if(t != N)
			yyerror("constdcl cannot have type without expr");
		cl = lastconst;
		t = lasttype;
	} else {
		lastconst = cl;
		lasttype = t;
	}
	cl = listtreecopy(cl);

	for(; vl; vl=vl->next) {
		if(cl == nil) {
			yyerror("missing expr in const dcl");
			break;
		}
		c = cl->n;
		cl = cl->next;

		v = vl->n;
		v->op = OLITERAL;
		declare(v, dclcontext);

		v->ntype = t;
		v->defn = c;

		vv = list(vv, nod(ODCLCONST, v, N));
	}
	if(cl != nil)
		yyerror("extra expr in const dcl");
	iota += 1;
	return vv;
}

/*
 * this generates a new name node,
 * typically for labels or other one-off names.
 */
Node*
newname(Sym *s)
{
	Node *n;

	if(s == S)
		fatal("newname nil");

	n = nod(ONAME, N, N);
	n->sym = s;
	n->type = T;
	n->addable = 1;
	n->ullman = 1;
	n->xoffset = 0;
	return n;
}

/*
 * this generates a new name node for a name
 * being declared.  if at the top level, it might return
 * an ONONAME node created by an earlier reference.
 */
Node*
dclname(Sym *s)
{
	Node *n;

	// top-level name: might already have been
	// referred to, in which case s->def is already
	// set to an ONONAME.
	if(dclcontext == PEXTERN && s->block <= 1) {
		if(s->def == N)
			oldname(s);
		if(s->def->op == ONONAME)
			return s->def;
	}

	n = newname(s);
	n->op = ONONAME;	// caller will correct it
	return n;
}

Node*
typenod(Type *t)
{
	// if we copied another type with *t = *u
	// then t->nod might be out of date, so
	// check t->nod->type too
	if(t->nod == N || t->nod->type != t) {
		t->nod = nod(OTYPE, N, N);
		t->nod->type = t;
		t->nod->sym = t->sym;
	}
	return t->nod;
}


/*
 * this will return an old name
 * that has already been pushed on the
 * declaration list. a diagnostic is
 * generated if no name has been defined.
 */
Node*
oldname(Sym *s)
{
	Node *n;
	Node *c;

	n = s->def;
	if(n == N) {
		// maybe a top-level name will come along
		// to give this a definition later.
		n = newname(s);
		n->op = ONONAME;
		s->def = n;
	}
	if(n->oldref < 100)
		n->oldref++;
	if(curfn != nil && n->funcdepth > 0 && n->funcdepth != funcdepth && n->op == ONAME) {
		// inner func is referring to var in outer func.
		//
		// TODO(rsc): If there is an outer variable x and we
		// are parsing x := 5 inside the closure, until we get to
		// the := it looks like a reference to the outer x so we'll
		// make x a closure variable unnecessarily.
		if(n->closure == N || n->closure->funcdepth != funcdepth) {
			// create new closure var.
			c = nod(ONAME, N, N);
			c->sym = s;
			c->class = PPARAMREF;
			c->isddd = n->isddd;
			c->defn = n;
			c->addable = 0;
			c->ullman = 2;
			c->funcdepth = funcdepth;
			c->outer = n->closure;
			n->closure = c;
			c->closure = n;
			c->xoffset = 0;
			curfn->cvars = list(curfn->cvars, c);
		}
		// return ref to closure var, not original
		return n->closure;
	}
	return n;
}

/*
 * same for types
 */
Type*
newtype(Sym *s)
{
	Type *t;

	t = typ(TFORW);
	t->sym = s;
	t->type = T;
	return t;
}

/*
 * type check top level declarations
 */
void
dclchecks(void)
{
	NodeList *l;

	for(l=externdcl; l; l=l->next) {
		if(l->n->op != ONAME)
			continue;
		typecheck(&l->n, Erv);
	}
}

/*
 * := declarations
 */

static int
colasname(Node *n)
{
	// TODO(rsc): can probably simplify
	// once late-binding of names goes in
	switch(n->op) {
	case ONAME:
	case ONONAME:
	case OPACK:
	case OTYPE:
	case OLITERAL:
		return n->sym != S;
	}
	return 0;
}

void
colasdefn(NodeList *left, Node *defn)
{
	int nnew;
	NodeList *l;
	Node *n;

	nnew = 0;
	for(l=left; l; l=l->next) {
		n = l->n;
		if(isblank(n))
			continue;
		if(!colasname(n)) {
			yyerror("non-name %#N on left side of :=", n);
			continue;
		}
		if(n->sym->block == block)
			continue;

		// If we created an ONONAME just for this :=,
		// delete it, to avoid confusion with top-level imports.
		if(n->op == ONONAME && n->oldref < 100 && --n->oldref == 0)
			n->sym->def = N;

		nnew++;
		n = newname(n->sym);
		declare(n, dclcontext);
		n->defn = defn;
		defn->ninit = list(defn->ninit, nod(ODCL, n, N));
		l->n = n;
	}
	if(nnew == 0)
		yyerror("no new variables on left side of :=");
}

Node*
colas(NodeList *left, NodeList *right)
{
	Node *as;

	as = nod(OAS2, N, N);
	as->list = left;
	as->rlist = right;
	as->colas = 1;
	colasdefn(left, as);

	// make the tree prettier; not necessary
	if(count(left) == 1 && count(right) == 1) {
		as->left = as->list->n;
		as->right = as->rlist->n;
		as->list = nil;
		as->rlist = nil;
		as->op = OAS;
	}

	return as;
}

/*
 * declare the function proper
 * and declare the arguments.
 * called in extern-declaration context
 * returns in auto-declaration context.
 */
void
funchdr(Node *n)
{

	if(n->nname != N) {
		n->nname->op = ONAME;
		declare(n->nname, PFUNC);
		n->nname->defn = n;
	}

	// change the declaration context from extern to auto
	if(funcdepth == 0 && dclcontext != PEXTERN)
		fatal("funchdr: dclcontext");

	dclcontext = PAUTO;
	markdcl();
	funcdepth++;

	n->outer = curfn;
	curfn = n;
	if(n->nname)
		funcargs(n->nname->ntype);
	else
		funcargs(n->ntype);
}

static void
funcargs(Node *nt)
{
	Node *n;
	NodeList *l;

	if(nt->op != OTFUNC)
		fatal("funcargs %O", nt->op);

	// declare the receiver and in arguments.
	// no n->defn because type checking of func header
	// will fill in the types before we can demand them.
	if(nt->left != N) {
		n = nt->left;
		if(n->op != ODCLFIELD)
			fatal("funcargs1 %O", n->op);
		if(n->left != N) {
			n->left->op = ONAME;
			n->left->ntype = n->right;
			declare(n->left, PPARAM);
		}
	}
	for(l=nt->list; l; l=l->next) {
		n = l->n;
		if(n->op != ODCLFIELD)
			fatal("funcargs2 %O", n->op);
		if(n->left != N) {
			n->left->op = ONAME;
			n->left->ntype = n->right;
			declare(n->left, PPARAM);
		}
	}

	// declare the out arguments.
	for(l=nt->rlist; l; l=l->next) {
		n = l->n;
		if(n->op != ODCLFIELD)
			fatal("funcargs3 %O", n->op);
		if(n->left != N) {
			n->left->op = ONAME;
			n->left->ntype = n->right;
			declare(n->left, PPARAMOUT);
		}
	}
}

/*
 * finish the body.
 * called in auto-declaration context.
 * returns in extern-declaration context.
 */
void
funcbody(Node *n)
{
	// change the declaration context from auto to extern
	if(dclcontext != PAUTO)
		fatal("funcbody: dclcontext");
	popdcl();
	funcdepth--;
	curfn = n->outer;
	n->outer = N;
	if(funcdepth == 0)
		dclcontext = PEXTERN;
}

/*
 * new type being defined with name s.
 */
Node*
typedcl0(Sym *s)
{
	Node *n;

	n = dclname(s);
	n->op = OTYPE;
	declare(n, dclcontext);
	return n;
}

/*
 * node n, which was returned by typedcl0
 * is being declared to have uncompiled type t.
 * return the ODCLTYPE node to use.
 */
Node*
typedcl1(Node *n, Node *t, int local)
{
	n->ntype = t;
	n->local = local;
	return nod(ODCLTYPE, n, N);
}

/*
 * typedcl1 but during imports
 */
void
typedcl2(Type *pt, Type *t)
{
	Node *n;

	if(pt->etype == TFORW)
		goto ok;
	if(!cvttype(pt, t))
		yyerror("inconsistent definition for type %S during import\n\t%lT\n\t%lT", pt->sym, pt, t);
	return;

ok:
	n = pt->nod;
	*pt = *t;
	pt->method = nil;
	pt->nod = n;
	pt->sym = n->sym;
	pt->sym->lastlineno = parserline();
	declare(n, PEXTERN);

	checkwidth(pt);
}

/*
 * structs, functions, and methods.
 * they don't belong here, but where do they belong?
 */


/*
 * turn a parsed struct into a type
 */
Type**
stotype(NodeList *l, int et, Type **t)
{
	Type *f, *t1, *t2, **t0;
	Strlit *note;
	int lno;
	NodeList *init;
	Node *n;
	char *what;

	t0 = t;
	init = nil;
	lno = lineno;
	what = "field";
	if(et == TINTER)
		what = "method";

	for(; l; l=l->next) {
		n = l->n;
		lineno = n->lineno;
		note = nil;

		if(n->op != ODCLFIELD)
			fatal("stotype: oops %N\n", n);
		if(n->right != N) {
			if(et == TINTER && n->left != N) {
				// queue resolution of method type for later.
				// right now all we need is the name list.
				// avoids cycles for recursive interface types.
				n->type = typ(TINTERMETH);
				n->type->nname = n->right;
				n->right = N;
				n->left->type = n->type;
				queuemethod(n);
			} else {
				typecheck(&n->right, Etype);
				n->type = n->right->type;
				if(n->type == T) {
					*t0 = T;
					return t0;
				}
				if(n->left != N)
					n->left->type = n->type;
				n->right = N;
				if(n->embedded && n->type != T) {
					t1 = n->type;
					if(t1->sym == S && isptr[t1->etype])
						t1 = t1->type;
					if(isptr[t1->etype])
						yyerror("embedded type cannot be a pointer");
					else if(t1->etype == TFORW && t1->embedlineno == 0)
						t1->embedlineno = lineno;
				}
			}
		}

		if(n->type == T) {
			// assume error already printed
			continue;
		}

		switch(n->val.ctype) {
		case CTSTR:
			if(et != TSTRUCT)
				yyerror("interface method cannot have annotation");
			note = n->val.u.sval;
			break;
		default:
			if(et != TSTRUCT)
				yyerror("interface method cannot have annotation");
			else
				yyerror("field annotation must be string");
		case CTxxx:
			note = nil;
			break;
		}

		if(et == TINTER && n->left == N) {
			// embedded interface - inline the methods
			if(n->type->etype != TINTER) {
				if(n->type->etype == TFORW)
					yyerror("interface type loop involving %T", n->type);
				else
					yyerror("interface contains embedded non-interface %T", n->type);
				continue;
			}
			for(t1=n->type->type; t1!=T; t1=t1->down) {
				f = typ(TFIELD);
				f->type = t1->type;
				f->width = BADWIDTH;
				f->nname = newname(t1->sym);
				f->sym = t1->sym;
				for(t2=*t0; t2!=T; t2=t2->down) {
					if(t2->sym == f->sym) {
						yyerror("duplicate method %s", t2->sym->name);
						break;
					}
				}
				*t = f;
				t = &f->down;
			}
			continue;
		}

		f = typ(TFIELD);
		f->type = n->type;
		f->note = note;
		f->width = BADWIDTH;
		f->isddd = n->isddd;

		if(n->left != N && n->left->op == ONAME) {
			f->nname = n->left;
			f->embedded = n->embedded;
			f->sym = f->nname->sym;
			if(importpkg && !exportname(f->sym->name))
				f->sym = pkglookup(f->sym->name, structpkg);
			if(f->sym && !isblank(f->nname)) {
				for(t1=*t0; t1!=T; t1=t1->down) {
					if(t1->sym == f->sym) {
						yyerror("duplicate %s %s", what, t1->sym->name);
						break;
					}
				}
			}
		}

		*t = f;
		t = &f->down;
	}

	*t = T;
	lineno = lno;
	return t;
}

Type*
dostruct(NodeList *l, int et)
{
	Type *t;
	int funarg;

	/*
	 * convert a parsed id/type list into
	 * a type for struct/interface/arglist
	 */

	funarg = 0;
	if(et == TFUNC) {
		funarg = 1;
		et = TSTRUCT;
	}
	t = typ(et);
	t->funarg = funarg;
	stotype(l, et, &t->type);
	if(t->type == T && l != nil) {
		t->broke = 1;
		return t;
	}
	if(!funarg)
		checkwidth(t);
	return t;
}


Node*
embedded(Sym *s)
{
	Node *n;
	char *name;

	// Names sometimes have disambiguation junk
	// appended after a center dot.  Discard it when
	// making the name for the embedded struct field.
	enum { CenterDot = 0xB7 };
	name = s->name;
	if(utfrune(s->name, CenterDot)) {
		name = strdup(s->name);
		*utfrune(name, CenterDot) = 0;
	}

	n = newname(lookup(name));
	n = nod(ODCLFIELD, n, oldname(s));
	n->embedded = 1;
	return n;
}

/*
 * check that the list of declarations is either all anonymous or all named
 */

static Node*
findtype(NodeList *l)
{
	for(; l; l=l->next)
		if(l->n->op == OKEY)
			return l->n->right;
	return N;
}

NodeList*
checkarglist(NodeList *all, int input)
{
	int named;
	Node *n, *t, *nextt;
	NodeList *l;

	named = 0;
	for(l=all; l; l=l->next) {
		if(l->n->op == OKEY) {
			named = 1;
			break;
		}
	}
	if(named) {
		n = N;
		for(l=all; l; l=l->next) {
			n = l->n;
			if(n->op != OKEY && n->sym == S) {
				yyerror("mixed named and unnamed function parameters");
				break;
			}
		}
		if(l == nil && n != N && n->op != OKEY)
			yyerror("final function parameter must have type");
	}

	nextt = nil;
	for(l=all; l; l=l->next) {
		// can cache result from findtype to avoid
		// quadratic behavior here, but unlikely to matter.
		n = l->n;
		if(named) {
			if(n->op == OKEY) {
				t = n->right;
				n = n->left;
				nextt = nil;
			} else {
				if(nextt == nil)
					nextt = findtype(l);
				t = nextt;
			}
		} else {
			t = n;
			n = N;
		}
		if(isblank(n))
			n = N;
		if(n != N && n->sym == S) {
			t = n;
			n = N;
		}
		if(n != N)
			n = newname(n->sym);
		n = nod(ODCLFIELD, n, t);
		if(n->right != N && n->right->op == ODDD) {
			if(!input)
				yyerror("cannot use ... in output argument list");
			else if(l->next != nil)
				yyerror("can only use ... as final argument in list");
			if(n->right->left == N) {
				// TODO(rsc): Delete with DDD cleanup.
				n->right->op = OTYPE;
				n->right->type = typ(TINTER);
			} else {
				n->right->op = OTARRAY;
				n->right->right = n->right->left;
				n->right->left = N;
			}
			n->isddd = 1;
			if(n->left != N)
				n->left->isddd = 1;
		}
		l->n = n;
	}
	return all;
}


Node*
fakethis(void)
{
	Node *n;

	n = nod(ODCLFIELD, N, typenod(ptrto(typ(TSTRUCT))));
	return n;
}

/*
 * Is this field a method on an interface?
 * Those methods have an anonymous
 * *struct{} as the receiver.
 * (See fakethis above.)
 */
int
isifacemethod(Type *f)
{
	Type *rcvr;
	Type *t;

	rcvr = getthisx(f)->type;
	if(rcvr->sym != S)
		return 0;
	t = rcvr->type;
	if(!isptr[t->etype])
		return 0;
	t = t->type;
	if(t->sym != S || t->etype != TSTRUCT || t->type != T)
		return 0;
	return 1;
}

/*
 * turn a parsed function declaration
 * into a type
 */
Type*
functype(Node *this, NodeList *in, NodeList *out)
{
	Type *t;
	NodeList *rcvr;

	t = typ(TFUNC);

	rcvr = nil;
	if(this)
		rcvr = list1(this);
	t->type = dostruct(rcvr, TFUNC);
	t->type->down = dostruct(out, TFUNC);
	t->type->down->down = dostruct(in, TFUNC);

	if(this)
		t->thistuple = 1;
	t->outtuple = count(out);
	t->intuple = count(in);
	t->outnamed = t->outtuple > 0 && out->n->left != N;

	return t;
}

int
methcmp(Type *t1, Type *t2)
{
	if(t1->etype != TFUNC)
		return 0;
	if(t2->etype != TFUNC)
		return 0;

	t1 = t1->type->down;	// skip this arg
	t2 = t2->type->down;	// skip this arg
	for(;;) {
		if(t1 == t2)
			break;
		if(t1 == T || t2 == T)
			return 0;
		if(t1->etype != TSTRUCT || t2->etype != TSTRUCT)
			return 0;

		if(!eqtype(t1->type, t2->type))
			return 0;

		t1 = t1->down;
		t2 = t2->down;
	}
	return 1;
}

Sym*
methodsym(Sym *nsym, Type *t0)
{
	Sym *s;
	char buf[NSYMB];
	Type *t;

	t = t0;
	if(t == T)
		goto bad;
	s = t->sym;
	if(s == S) {
		if(!isptr[t->etype])
			goto bad;
		t = t->type;
		if(t == T)
			goto bad;
		s = t->sym;
		if(s == S)
			goto bad;
	}

	// if t0 == *t and t0 has a sym,
	// we want to see *t, not t0, in the method name.
	if(t != t0 && t0->sym)
		t0 = ptrto(t);

	snprint(buf, sizeof(buf), "%#hT·%s", t0, nsym->name);
	return pkglookup(buf, s->pkg);

bad:
	yyerror("illegal receiver type: %T", t0);
	return S;
}

Node*
methodname(Node *n, Type *t)
{
	Sym *s;

	s = methodsym(n->sym, t);
	if(s == S)
		return n;
	return newname(s);
}

Node*
methodname1(Node *n, Node *t)
{
	char *star;
	char buf[NSYMB];

	star = "";
	if(t->op == OIND) {
		star = "*";
		t = t->left;
	}
	if(t->sym == S || isblank(n))
		return newname(n->sym);
	snprint(buf, sizeof(buf), "%s%S·%S", star, t->sym, n->sym);
	return newname(pkglookup(buf, t->sym->pkg));
}

/*
 * add a method, declared as a function,
 * n is fieldname, pa is base type, t is function type
 */
void
addmethod(Sym *sf, Type *t, int local)
{
	Type *f, *d, *pa;
	Node *n;

	pa = nil;

	// get field sym
	if(sf == S)
		fatal("no method symbol");

	// get parent type sym
	pa = getthisx(t)->type;	// ptr to this structure
	if(pa == T) {
		yyerror("missing receiver");
		return;
	}

	pa = pa->type;
	f = methtype(pa);
	if(f == T) {
		yyerror("invalid receiver type %T", pa);
		return;
	}

	pa = f;
	if(importpkg && !exportname(sf->name))
		sf = pkglookup(sf->name, importpkg);

	n = nod(ODCLFIELD, newname(sf), N);
	n->type = t;

	d = T;	// last found
	for(f=pa->method; f!=T; f=f->down) {
		d = f;
		if(f->etype != TFIELD)
			fatal("addmethod: not TFIELD: %N", f);
		if(strcmp(sf->name, f->sym->name) != 0)
			continue;
		if(!eqtype(t, f->type))
			yyerror("method redeclared: %T.%S\n\t%T\n\t%T", pa, sf, f->type, t);
		return;
	}

	if(local && !pa->local) {
		// defining method on non-local type.
		yyerror("cannot define new methods on non-local type %T", pa);
		return;
	}

	if(d == T)
		stotype(list1(n), 0, &pa->method);
	else
		stotype(list1(n), 0, &d->down);
	return;
}

void
funccompile(Node *n)
{
	stksize = BADWIDTH;
	maxarg = 0;

	if(n->type == T) {
		if(nerrors == 0)
			fatal("funccompile missing type");
		return;
	}

	// assign parameter offsets
	checkwidth(n->type);

	if(curfn)
		fatal("funccompile %S inside %S", n->nname->sym, curfn->nname->sym);
	curfn = n;
	typechecklist(n->nbody, Etop);
	curfn = nil;

	stksize = 0;
	dclcontext = PAUTO;
	funcdepth = n->funcdepth + 1;
	compile(n);
	curfn = nil;
	funcdepth = 0;
	dclcontext = PEXTERN;
}

