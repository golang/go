// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include	<u.h>
#include	<libc.h>
#include	"go.h"
#include	"y.tab.h"

static	void	funcargs(Node*);
static	void	funcargs2(Type*);

static int
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

static void
dcopy(Sym *a, Sym *b)
{
	a->pkg = b->pkg;
	a->name = b->name;
	a->def = b->def;
	a->block = b->block;
	a->lastlineno = b->lastlineno;
}

static Sym*
push(void)
{
	Sym *d;

	d = mal(sizeof(*d));
	d->lastlineno = lineno;
	d->link = dclstack;
	dclstack = d;
	return d;
}

static Sym*
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
	int lno;

//	if(dflag())
//		print("revert\n");

	for(d=dclstack; d!=S; d=d->link) {
		if(d->name == nil)
			break;
		s = pkglookup(d->name, d->pkg);
		lno = s->lastlineno;
		dcopy(s, d);
		d->lastlineno = lno;
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
	// pop the old marker and push a new one
	// (cannot reuse the existing one)
	// because we use the markers to identify blocks
	// for the goto restriction checks.
	popdcl();
	markdcl();
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

	USED(st);

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
	Strlit *pkgstr;
	int line1, line2;

	if(s->lastlineno == 0) {
		pkgstr = s->origpkg ? s->origpkg->path : s->pkg->path;
		yyerror("%S redeclared %s\n"
			"\tprevious declaration during import \"%Z\"",
			s, where, pkgstr);
	} else {
		line1 = parserline();
		line2 = s->lastlineno;
		
		// When an import and a declaration collide in separate files,
		// present the import as the "redeclared", because the declaration
		// is visible where the import is, but not vice versa.
		// See issue 4510.
		if(s->def == N) {
			line2 = line1;
			line1 = s->lastlineno;
		}

		yyerrorl(line1, "%S redeclared %s\n"
			"\tprevious declaration at %L",
			s, where, line2);
	}
}

static int vargen;

/*
 * declare individual names - var, typ, const
 */
void
declare(Node *n, int ctxt)
{
	Sym *s;
	int gen;
	static int typegen;
	
	if(ctxt == PDISCARD)
		return;

	if(isblank(n))
		return;

	n->lineno = parserline();
	s = n->sym;

	// kludgy: typecheckok means we're past parsing.  Eg genwrapper may declare out of package names later.
	if(importpkg == nil && !typecheckok && s->pkg != localpkg)
		yyerror("cannot declare name %S", s);

	if(ctxt == PEXTERN && strcmp(s->name, "init") == 0)
		yyerror("cannot declare init - must be func", s);

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
		else if(n->op == ONAME && ctxt == PAUTO && strstr(s->name, "·") == nil)
			gen = ++vargen;
		pushdcl(s);
		n->curfn = curfn;
	}
	if(ctxt == PAUTO)
		n->xoffset = 0;

	if(s->block == block) {
		// functype will print errors about duplicate function arguments.
		// Don't repeat the error here.
		if(ctxt != PPARAM && ctxt != PPARAMOUT)
			redeclare(s, "in this block");
	}

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
 * being declared.
 */
Node*
dclname(Sym *s)
{
	Node *n;

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
		// walkdef will check s->def again once
		// all the input source has been processed.
		n = newname(s);
		n->op = ONONAME;
		n->iota = iota;	// save current iota value in const declarations
	}
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
			n->addrtaken = 1;
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
 * := declarations
 */

static int
colasname(Node *n)
{
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
	int nnew, nerr;
	NodeList *l;
	Node *n;

	nnew = 0;
	nerr = 0;
	for(l=left; l; l=l->next) {
		n = l->n;
		if(isblank(n))
			continue;
		if(!colasname(n)) {
			yyerrorl(defn->lineno, "non-name %N on left side of :=", n);
			nerr++;
			continue;
		}
		if(n->sym->block == block)
			continue;

		nnew++;
		n = newname(n->sym);
		declare(n, dclcontext);
		n->defn = defn;
		defn->ninit = list(defn->ninit, nod(ODCL, n, N));
		l->n = n;
	}
	if(nnew == 0 && nerr == 0)
		yyerrorl(defn->lineno, "no new variables on left side of :=");
}

Node*
colas(NodeList *left, NodeList *right, int32 lno)
{
	Node *as;

	as = nod(OAS2, N, N);
	as->list = left;
	as->rlist = right;
	as->colas = 1;
	as->lineno = lno;
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
 * declare the arguments in an
 * interface field declaration.
 */
void
ifacedcl(Node *n)
{
	if(n->op != ODCLFIELD || n->right == N)
		fatal("ifacedcl");

	dclcontext = PPARAM;
	markdcl();
	funcdepth++;
	n->outer = curfn;
	curfn = n;
	funcargs(n->right);

	// funcbody is normally called after the parser has
	// seen the body of a function but since an interface
	// field declaration does not have a body, we must
	// call it now to pop the current declaration context.
	dclcontext = PAUTO;
	funcbody(n);
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
	else if (n->ntype)
		funcargs(n->ntype);
	else
		funcargs2(n->type);
}

static void
funcargs(Node *nt)
{
	Node *n, *nn;
	NodeList *l;
	int gen;

	if(nt->op != OTFUNC)
		fatal("funcargs %O", nt->op);

	// re-start the variable generation number
	// we want to use small numbers for the return variables,
	// so let them have the chunk starting at 1.
	vargen = count(nt->rlist);

	// declare the receiver and in arguments.
	// no n->defn because type checking of func header
	// will not fill in the types until later
	if(nt->left != N) {
		n = nt->left;
		if(n->op != ODCLFIELD)
			fatal("funcargs receiver %O", n->op);
		if(n->left != N) {
			n->left->op = ONAME;
			n->left->ntype = n->right;
			declare(n->left, PPARAM);
			if(dclcontext == PAUTO)
				n->left->vargen = ++vargen;
		}
	}
	for(l=nt->list; l; l=l->next) {
		n = l->n;
		if(n->op != ODCLFIELD)
			fatal("funcargs in %O", n->op);
		if(n->left != N) {
			n->left->op = ONAME;
			n->left->ntype = n->right;
			declare(n->left, PPARAM);
			if(dclcontext == PAUTO)
				n->left->vargen = ++vargen;
		}
	}

	// declare the out arguments.
	gen = count(nt->list);
	int i = 0;
	for(l=nt->rlist; l; l=l->next) {
		n = l->n;

		if(n->op != ODCLFIELD)
			fatal("funcargs out %O", n->op);

		if(n->left == N) {
			// give it a name so escape analysis has nodes to work with
			snprint(namebuf, sizeof(namebuf), "~anon%d", gen++);
			n->left = newname(lookup(namebuf));
			// TODO: n->left->missing = 1;
		} 

		n->left->op = ONAME;

		if(isblank(n->left)) {
			// Give it a name so we can assign to it during return.
			// preserve the original in ->orig
			nn = nod(OXXX, N, N);
			*nn = *n->left;
			n->left = nn;
			
			snprint(namebuf, sizeof(namebuf), "~anon%d", gen++);
			n->left->sym = lookup(namebuf);
		}

		n->left->ntype = n->right;
		declare(n->left, PPARAMOUT);
		if(dclcontext == PAUTO)
			n->left->vargen = ++i;
	}
}

/*
 * Same as funcargs, except run over an already constructed TFUNC.
 * This happens during import, where the hidden_fndcl rule has
 * used functype directly to parse the function's type.
 */
static void
funcargs2(Type *t)
{
	Type *ft;
	Node *n;

	if(t->etype != TFUNC)
		fatal("funcargs2 %T", t);
	
	if(t->thistuple)
		for(ft=getthisx(t)->type; ft; ft=ft->down) {
			if(!ft->nname || !ft->nname->sym)
				continue;
			n = ft->nname;  // no need for newname(ft->nname->sym)
			n->type = ft->type;
			declare(n, PPARAM);
		}

	if(t->intuple)
		for(ft=getinargx(t)->type; ft; ft=ft->down) {
			if(!ft->nname || !ft->nname->sym)
				continue;
			n = ft->nname;
			n->type = ft->type;
			declare(n, PPARAM);
		}

	if(t->outtuple)
		for(ft=getoutargx(t)->type; ft; ft=ft->down) {
			if(!ft->nname || !ft->nname->sym)
				continue;
			n = ft->nname;
			n->type = ft->type;
			declare(n, PPARAMOUT);
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

	n = newname(s);
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
 * structs, functions, and methods.
 * they don't belong here, but where do they belong?
 */

static void
checkembeddedtype(Type *t)
{
	if (t == T)
		return;

	if(t->sym == S && isptr[t->etype]) {
		t = t->type;
		if(t->etype == TINTER)
			yyerror("embedded type cannot be a pointer to interface");
	}
	if(isptr[t->etype])
		yyerror("embedded type cannot be a pointer");
	else if(t->etype == TFORW && t->embedlineno == 0)
		t->embedlineno = lineno;
}

static Type*
structfield(Node *n)
{
	Type *f;
	int lno;

	lno = lineno;
	lineno = n->lineno;

	if(n->op != ODCLFIELD)
		fatal("structfield: oops %N\n", n);

	f = typ(TFIELD);
	f->isddd = n->isddd;

	if(n->right != N) {
		typecheck(&n->right, Etype);
		n->type = n->right->type;
		if(n->left != N)
			n->left->type = n->type;
		if(n->embedded)
			checkembeddedtype(n->type);
	}
	n->right = N;
		
	f->type = n->type;
	if(f->type == T)
		f->broke = 1;

	switch(n->val.ctype) {
	case CTSTR:
		f->note = n->val.u.sval;
		break;
	default:
		yyerror("field annotation must be string");
		// fallthrough
	case CTxxx:
		f->note = nil;
		break;
	}

	if(n->left && n->left->op == ONAME) {
		f->nname = n->left;
		f->embedded = n->embedded;
		f->sym = f->nname->sym;
	}

	lineno = lno;
	return f;
}

static uint32 uniqgen;

static void
checkdupfields(Type *t, char* what)
{
	int lno;

	lno = lineno;

	for( ; t; t=t->down) {
		if(t->sym && t->nname && !isblank(t->nname)) {
			if(t->sym->uniqgen == uniqgen) {
				lineno = t->nname->lineno;
				yyerror("duplicate %s %s", what, t->sym->name);
			} else
				t->sym->uniqgen = uniqgen;
		}
	}

	lineno = lno;
}

/*
 * convert a parsed id/type list into
 * a type for struct/interface/arglist
 */
Type*
tostruct(NodeList *l)
{
	Type *t, *f, **tp;
	t = typ(TSTRUCT);

	for(tp = &t->type; l; l=l->next) {
		f = structfield(l->n);

		*tp = f;
		tp = &f->down;
	}

	for(f=t->type; f && !t->broke; f=f->down)
		if(f->broke)
			t->broke = 1;

	uniqgen++;
	checkdupfields(t->type, "field");

	if (!t->broke)
		checkwidth(t);

	return t;
}

static Type*
tofunargs(NodeList *l)
{
	Type *t, *f, **tp;

	t = typ(TSTRUCT);
	t->funarg = 1;

	for(tp = &t->type; l; l=l->next) {
		f = structfield(l->n);
		f->funarg = 1;

		// esc.c needs to find f given a PPARAM to add the tag.
		if(l->n->left && l->n->left->class == PPARAM)
			l->n->left->paramfld = f;

		*tp = f;
		tp = &f->down;
	}

	for(f=t->type; f && !t->broke; f=f->down)
		if(f->broke)
			t->broke = 1;

	return t;
}

static Type*
interfacefield(Node *n)
{
	Type *f;
	int lno;

	lno = lineno;
	lineno = n->lineno;

	if(n->op != ODCLFIELD)
		fatal("interfacefield: oops %N\n", n);

	if (n->val.ctype != CTxxx)
		yyerror("interface method cannot have annotation");

	f = typ(TFIELD);
	f->isddd = n->isddd;
	
	if(n->right != N) {
		if(n->left != N) {
			// queue resolution of method type for later.
			// right now all we need is the name list.
			// avoids cycles for recursive interface types.
			n->type = typ(TINTERMETH);
			n->type->nname = n->right;
			n->left->type = n->type;
			queuemethod(n);

			if(n->left->op == ONAME) {
				f->nname = n->left;
				f->embedded = n->embedded;
				f->sym = f->nname->sym;
				if(importpkg && !exportname(f->sym->name))
					f->sym = pkglookup(f->sym->name, structpkg);
			}

		} else {

			typecheck(&n->right, Etype);
			n->type = n->right->type;

			if(n->embedded)
				checkembeddedtype(n->type);

			if(n->type)
				switch(n->type->etype) {
				case TINTER:
					break;
				case TFORW:
					yyerror("interface type loop involving %T", n->type);
					f->broke = 1;
					break;
				default:
					yyerror("interface contains embedded non-interface %T", n->type);
					f->broke = 1;
					break;
				}
		}
	}

	n->right = N;
	
	f->type = n->type;
	if(f->type == T)
		f->broke = 1;
	
	lineno = lno;
	return f;
}

Type*
tointerface(NodeList *l)
{
	Type *t, *f, **tp, *t1;

	t = typ(TINTER);

	tp = &t->type;
	for(; l; l=l->next) {
		f = interfacefield(l->n);

		if (l->n->left == N && f->type->etype == TINTER) {
			// embedded interface, inline methods
			for(t1=f->type->type; t1; t1=t1->down) {
				f = typ(TFIELD);
				f->type = t1->type;
				f->broke = t1->broke;
				f->sym = t1->sym;
				if(f->sym)
					f->nname = newname(f->sym);
				*tp = f;
				tp = &f->down;
			}
		} else {
			*tp = f;
			tp = &f->down;
		}
	}

	for(f=t->type; f && !t->broke; f=f->down)
		if(f->broke)
			t->broke = 1;

	uniqgen++;
	checkdupfields(t->type, "method");
	t = sortinter(t);
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

	if(exportname(name))
		n = newname(lookup(name));
	else if(s->pkg == builtinpkg && importpkg != nil)
		// The name of embedded builtins during imports belongs to importpkg.
		n = newname(pkglookup(name, importpkg));
	else
		n = newname(pkglookup(name, s->pkg));
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

		// during import l->n->op is OKEY, but l->n->left->sym == S
		// means it was a '?', not that it was
		// a lone type This doesn't matter for the exported
		// declarations, which are parsed by rules that don't
		// use checkargs, but can happen for func literals in
		// the inline bodies.
		// TODO(rsc) this can go when typefmt case TFIELD in exportmode fmt.c prints _ instead of ?
		if(importpkg && n->sym == S)
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
			n->right->op = OTARRAY;
			n->right->right = n->right->left;
			n->right->left = N;
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
	Sym *s;

	t = typ(TFUNC);

	rcvr = nil;
	if(this)
		rcvr = list1(this);
	t->type = tofunargs(rcvr);
	t->type->down = tofunargs(out);
	t->type->down->down = tofunargs(in);

	uniqgen++;
	checkdupfields(t->type->type, "argument");
	checkdupfields(t->type->down->type, "argument");
	checkdupfields(t->type->down->down->type, "argument");

	if (t->type->broke || t->type->down->broke || t->type->down->down->broke)
		t->broke = 1;

	if(this)
		t->thistuple = 1;
	t->outtuple = count(out);
	t->intuple = count(in);
	t->outnamed = 0;
	if(t->outtuple > 0 && out->n->left != N && out->n->left->orig != N) {
		s = out->n->left->orig->sym;
		if(s != S && s->name[0] != '~')
			t->outnamed = 1;
	}

	return t;
}

Sym*
methodsym(Sym *nsym, Type *t0, int iface)
{
	Sym *s;
	char *p;
	Type *t;
	char *suffix;
	Pkg *spkg;
	static Pkg *toppkg;

	t = t0;
	if(t == T)
		goto bad;
	s = t->sym;
	if(s == S && isptr[t->etype]) {
		t = t->type;
		if(t == T)
			goto bad;
		s = t->sym;
	}
	spkg = nil;
	if(s != S)
		spkg = s->pkg;

	// if t0 == *t and t0 has a sym,
	// we want to see *t, not t0, in the method name.
	if(t != t0 && t0->sym)
		t0 = ptrto(t);

	suffix = "";
	if(iface) {
		dowidth(t0);
		if(t0->width < types[tptr]->width)
			suffix = "·i";
	}
	if((spkg == nil || nsym->pkg != spkg) && !exportname(nsym->name)) {
		if(t0->sym == S && isptr[t0->etype])
			p = smprint("(%-hT).%s.%s%s", t0, nsym->pkg->prefix, nsym->name, suffix);
		else
			p = smprint("%-hT.%s.%s%s", t0, nsym->pkg->prefix, nsym->name, suffix);
	} else {
		if(t0->sym == S && isptr[t0->etype])
			p = smprint("(%-hT).%s%s", t0, nsym->name, suffix);
		else
			p = smprint("%-hT.%s%s", t0, nsym->name, suffix);
	}
	if(spkg == nil) {
		if(toppkg == nil)
			toppkg = mkpkg(strlit("go"));
		spkg = toppkg;
	}
	s = pkglookup(p, spkg);
	free(p);
	return s;

bad:
	yyerror("illegal receiver type: %T", t0);
	return S;
}

Node*
methodname(Node *n, Type *t)
{
	Sym *s;

	s = methodsym(n->sym, t, 0);
	if(s == S)
		return n;
	return newname(s);
}

Node*
methodname1(Node *n, Node *t)
{
	char *star;
	char *p;

	star = nil;
	if(t->op == OIND) {
		star = "*";
		t = t->left;
	}
	if(t->sym == S || isblank(n))
		return newname(n->sym);

	if(star)
		p = smprint("(%s%S).%S", star, t->sym, n->sym);
	else
		p = smprint("%S.%S", t->sym, n->sym);

	if(exportname(t->sym->name))
		n = newname(lookup(p));
	else
		n = newname(pkglookup(p, t->sym->pkg));
	free(p);
	return n;
}

/*
 * add a method, declared as a function,
 * n is fieldname, pa is base type, t is function type
 */
void
addmethod(Sym *sf, Type *t, int local, int nointerface)
{
	Type *f, *d, *pa;
	Node *n;

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
	f = methtype(pa, 1);
	if(f == T) {
		t = pa;
		if(t == T) // rely on typecheck having complained before
			return;
		if(t != T) {
			if(isptr[t->etype]) {
				if(t->sym != S) {
					yyerror("invalid receiver type %T (%T is a pointer type)", pa, t);
					return;
				}
				t = t->type;
			}
			if(t->broke) // rely on typecheck having complained before
				return;
			if(t->sym == S) {
				yyerror("invalid receiver type %T (%T is an unnamed type)", pa, t);
				return;
			}
			if(isptr[t->etype]) {
				yyerror("invalid receiver type %T (%T is a pointer type)", pa, t);
				return;
			}
			if(t->etype == TINTER) {
				yyerror("invalid receiver type %T (%T is an interface type)", pa, t);
				return;
			}
		}
		// Should have picked off all the reasons above,
		// but just in case, fall back to generic error.
		yyerror("invalid receiver type %T (%lT / %lT)", pa, pa, t);
		return;
	}

	pa = f;
	if(pa->etype == TSTRUCT) {
		for(f=pa->type; f; f=f->down) {
			if(f->sym == sf) {
				yyerror("type %T has both field and method named %S", pa, sf);
				return;
			}
		}
	}

	if(local && !pa->local) {
		// defining method on non-local type.
		yyerror("cannot define new methods on non-local type %T", pa);
		return;
	}

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

	f = structfield(n);
	f->nointerface = nointerface;

	// during import unexported method names should be in the type's package
	if(importpkg && f->sym && !exportname(f->sym->name) && f->sym->pkg != structpkg)
		fatal("imported method name %+S in wrong package %s\n", f->sym, structpkg->name);

	if(d == T)
		pa->method = f;
	else
		d->down = f;
	return;
}

void
funccompile(Node *n, int isclosure)
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
	
	// record offset to actual frame pointer.
	// for closure, have to skip over leading pointers and PC slot.
	nodfp->xoffset = 0;
	if(isclosure) {
		NodeList *l;
		for(l=n->nname->ntype->list; l; l=l->next) {
			nodfp->xoffset += widthptr;
			if(l->n->left == N)	// found slot for PC
				break;
		}
	}

	if(curfn)
		fatal("funccompile %S inside %S", n->nname->sym, curfn->nname->sym);

	stksize = 0;
	dclcontext = PAUTO;
	funcdepth = n->funcdepth + 1;
	compile(n);
	curfn = nil;
	funcdepth = 0;
	dclcontext = PEXTERN;
}

Sym*
funcsym(Sym *s)
{
	char *p;
	Sym *s1;
	
	p = smprint("%s·f", s->name);
	s1 = pkglookup(p, s->pkg);
	free(p);
	if(s1->def == N) {
		s1->def = newname(s1);
		s1->def->shortname = newname(s);
		funcsyms = list(funcsyms, s1->def);
	}
	return s1;
}
