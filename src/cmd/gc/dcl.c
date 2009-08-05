// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include	"go.h"
#include	"y.tab.h"

int
dflag(void)
{
	if(!debug['d'])
		return 0;
	if(debug['y'])
		return 1;
	if(inimportsys)
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
	a->package = b->package;
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
		s = pkglookup(d->name, d->package);
		dcopy(s, d);
		if(dflag())
			print("\t%L pop %S\n", lineno, s);
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
		s = pkglookup(d->name, d->package);
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
		s = pkglookup(d->name, d->package);
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

/*
 * declare individual names - var, typ, const
 */
void
declare(Node *n, int ctxt)
{
	Sym *s;
	char *what;
	int gen;
	static int typegen, vargen;

	s = n->sym;
	gen = 0;
	if(ctxt == PEXTERN) {
		externdcl = list(externdcl, n);
	} else {
		if(autodcl != nil)
			autodcl = list(autodcl, n);
		if(n->op == OTYPE)
			gen = ++typegen;
		else if(n->op == ONAME)
			gen = ++vargen;
		pushdcl(s);
	}

	if(s->block == block) {
		what = "???";
		switch(n->op) {
		case ONAME:
			what = "variable";
			break;
		case OLITERAL:
			what = "constant";
			break;
		case OTYPE:
			what = "type";
			break;
		}

		yyerror("%s %S redeclared in this block", what, s);
		print("\tprevious declaration at %L\n", s->lastlineno);
	}
	s->block = block;
	s->lastlineno = lineno;
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
 * declare (possible list) n of type t.
 * append ODCL nodes to *init
 */
void
dodclvar(Node *n, Type *t, NodeList **init)
{
	if(n == N)
		return;

	if(t != T && (t->etype == TIDEAL || t->etype == TNIL))
		fatal("dodclvar %T", t);
	dowidth(t);

	// in case of type checking error,
	// use "undefined" type for variable type,
	// to avoid fatal in addvar.
	if(t == T)
		t = typ(TFORW);

	addvar(n, t, dclcontext);
	if(funcdepth > 0)
		*init = list(*init, nod(ODCL, n, N));
}

/*
 * introduce a type named n
 * but it is an unknown type for now
 */
Type*
dodcltype(Type *n)
{
	Sym *s;

	// if n has been forward declared,
	// use the Type* created then
	s = n->sym;
	if((funcdepth == 0 || s->block == block) && s->def != N && s->def->op == OTYPE) {
		switch(s->def->type->etype) {
		case TFORWSTRUCT:
		case TFORWINTER:
			n = s->def->type;
			if(s->block != block) {
				// completing forward struct from other file
				externdcl = list(externdcl, typenod(n));
			}
			goto found;
		}
	}

	// otherwise declare a new type
	addtyp(n, dclcontext);

found:
	n->local = 1;
	autoexport(typenod(n), dclcontext);
	return n;
}

/*
 * now we know what n is: it's t
 */
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

	case TFORWSTRUCT:
		if(t->etype != TSTRUCT) {
			yyerror("%T forward declared as struct", n);
			return;
		}
		n->local = 1;
		break;

	case TFORWINTER:
		if(t->etype != TINTER) {
			yyerror("%T forward declared as interface", n);
			return;
		}
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

	// catch declaration of incomplete type
	switch(n->etype) {
	case TFORWSTRUCT:
	case TFORWINTER:
		break;
	default:
		checkwidth(n);
	}

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
	Node *v, *e;
	NodeList *init;
	Sym *s;

	init = nil;
	doexpr = el != nil;
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
		s = v->sym;

		v->op = ONAME;
		declare(v, dclcontext);
		v->ntype = t;

		if(e != N || funcdepth > 0) {
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

	vv = vl;
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
	if(dclcontext == PEXTERN && s->block == 0) {
		// toss predefined name like "close"
		// TODO(rsc): put close in at the end.
		if(s->def != N && s->def->etype)
			s->def = N;
		if(s->def == N)
			oldname(s);
		return s->def;
	}

	n = newname(s);
	n->op = ONONAME;	// caller will correct it
	return n;
}

Node*
typenod(Type *t)
{
	if(t->nod == N) {
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
	if(n->funcdepth > 0 && n->funcdepth != funcdepth && n->op == ONAME) {
		// inner func is referring to var
		// in outer func.
		if(n->closure == N || n->closure->funcdepth != funcdepth) {
			typecheck(&n, Erv);
			// create new closure var.
			c = nod(ONAME, N, N);
			c->sym = s;
			c->class = PPARAMREF;
			c->type = n->type;
			c->addable = 0;
			c->ullman = 2;
			c->funcdepth = funcdepth;
			c->outer = n->closure;
			n->closure = c;
			c->closure = n;
			if(funclit != N)
				funclit->cvars = list(funclit->cvars, c);
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
 * structs, functions, and methods.
 * they don't belong here, but where do they belong?
 */


/*
 * turn a parsed struct into a type
 */
Type**
stotype(NodeList *l, int et, Type **t)
{
	Type *f, *t1;
	Strlit *note;
	int lno;
	NodeList *init;
	Node *n;

	init = nil;
	lno = lineno;
	for(; l; l=l->next) {
		n = l->n;
		lineno = n->lineno;
		note = nil;

		if(n->op != ODCLFIELD)
			fatal("stotype: oops %N\n", n);
		if(n->right != N) {
			typecheck(&n->right, Etype);
			n->type = n->right->type;
			n->right = N;
			if(n->embedded && n->type != T) {
				t1 = n->type;
				if(t1->sym == S && isptr[t1->etype])
					t1 = t1->type;
				if(t1 != T && isptr[t1->etype])
					yyerror("embedded type cannot be a pointer");
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
				yyerror("interface contains embedded non-interface %T", t);
				continue;
			}
			for(t1=n->type->type; t1!=T; t1=t1->down) {
				// TODO(rsc): Is this really an error?
				if(strcmp(t1->sym->package, package) != 0)
					yyerror("embedded interface contains unexported method %S", t1->sym);
				f = typ(TFIELD);
				f->type = t1->type;
				f->width = BADWIDTH;
				f->nname = newname(t1->sym);
				f->sym = t1->sym;
				*t = f;
				t = &f->down;
			}
			continue;
		}

		f = typ(TFIELD);
		f->type = n->type;
		f->note = note;
		f->width = BADWIDTH;

		if(n->left != N && n->left->op == ONAME) {
			f->nname = n->left;
			f->embedded = n->embedded;
			f->sym = f->nname->sym;
			if(pkgimportname != S && !exportname(f->sym->name))
				f->sym = pkglookup(f->sym->name, structpkg);
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
	n = nod(ODCLFIELD, n, N);
	n->embedded = 1;
	if(s == S)
		return n;
	n->right = oldname(s);
	return n;
}

static Node*
findtype(NodeList *l)
{
	for(; l; l=l->next)
		if(l->n->op == OKEY)
			return l->n->right;
	return N;
}

static Node*
xanondcl(Node *nt)
{
	Node *n;
	Type *t;

	typecheck(&nt, Etype);
	t = nt->type;
	if(nt->op != OTYPE) {
		yyerror("%S is not a type", nt->sym);
		t = types[TINT32];
	}
	n = nod(ODCLFIELD, N, N);
	n->type = t;
	return n;
}

static Node*
namedcl(Node *nn, Node *nt)
{
	Node *n;
	Type *t;

	if(nn->op == OKEY)
		nn = nn->left;
	if(nn->sym == S) {
		typecheck(&nn, Etype);
		yyerror("cannot mix anonymous %T with named arguments", nn->type);
		return xanondcl(nn);
	}
	t = types[TINT32];
	if(nt == N)
		yyerror("missing type for argument %S", nn->sym);
	else {
		typecheck(&nt, Etype);
		if(nt->op != OTYPE)
			yyerror("%S is not a type", nt->sym);
		else
			t = nt->type;
	}
	n = nod(ODCLFIELD, newname(nn->sym), N);
	n->type = t;
	return n;
}

/*
 * check that the list of declarations is either all anonymous or all named
 */
NodeList*
checkarglist(NodeList *all)
{
	int named;
	Node *r;
	NodeList *l;

	named = 0;
	for(l=all; l; l=l->next) {
		if(l->n->op == OKEY) {
			named = 1;
			break;
		}
	}

	for(l=all; l; l=l->next) {
		if(named)
			l->n = namedcl(l->n, findtype(l));
		else
			l->n = xanondcl(l->n);
		if(l->next != nil) {
			r = l->n;
			if(r != N && r->type != T && r->type->etype == TDDD)
				yyerror("only last argument can have type ...");
		}
	}
	return all;
}


Node*
fakethis(void)
{
	Node *n;

	n = nod(ODCLFIELD, N, N);
	n->type = ptrto(typ(TSTRUCT));
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

	rcvr = getthisx(f->type)->type;
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

	checkwidth(t);
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
//print("methodname %s\n", buf);
	return pkglookup(buf, s->package);

bad:
	yyerror("illegal <this> type: %T", t);
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

/*
 * add a method, declared as a function,
 * n is fieldname, pa is base type, t is function type
 */
void
addmethod(Node *n, Type *t, int local)
{
	Type *f, *d, *pa;
	Sym *sf;

	pa = nil;
	sf = nil;

	// get field sym
	if(n == N)
		goto bad;
	if(n->op != ONAME)
		goto bad;
	sf = n->sym;
	if(sf == S)
		goto bad;

	// get parent type sym
	pa = *getthis(t);	// ptr to this structure
	if(pa == T)
		goto bad;
	pa = pa->type;		// ptr to this field
	if(pa == T)
		goto bad;
	pa = pa->type;		// ptr to this type
	if(pa == T)
		goto bad;

	f = methtype(pa);
	if(f == T)
		goto bad;

	pa = f;
	if(pkgimportname != S && !exportname(sf->name))
		sf = pkglookup(sf->name, pkgimportname->name);

	n = nod(ODCLFIELD, newname(sf), N);
	n->type = t;

	d = T;	// last found
	for(f=pa->method; f!=T; f=f->down) {
		d = f;
		if(f->etype != TFIELD)
			fatal("addmethod: not TFIELD: %N", f);
		if(strcmp(sf->name, f->sym->name) != 0)
			continue;
		if(!eqtype(t, f->type)) {
			yyerror("method redeclared: %T.%S", pa, sf);
			print("\t%T\n\t%T\n", f->type, t);
		}
		return;
	}

	if(local && !pa->local) {
		// defining method on non-local type.
		// method must have been forward declared
		// elsewhere, i.e. where the type was.
		yyerror("cannot define new methods on non-local type %T", pa);
		return;
	}

	if(d == T)
		stotype(list1(n), 0, &pa->method);
	else
		stotype(list1(n), 0, &d->down);
	return;

bad:
	yyerror("invalid receiver type %T", pa);
}

/*
 * declare the function proper.
 * and declare the arguments
 * called in extern-declaration context
 * returns in auto-declaration context.
 */
void
funchdr(Node *n)
{
	Node *on;
	Sym *s;

	s = n->nname->sym;
	on = s->def;
	if(on != N && (on->op != ONAME || on->builtin))
		on = N;

	// check for same types
	if(on != N) {
		if(eqtype(n->type, on->type)) {
			if(!eqargs(n->type, on->type)) {
				yyerror("function arg names changed: %S", s);
				print("\t%T\n\t%T\n", on->type, n->type);
			}
		} else {
			yyerror("function redeclared: %S", s);
			print("\t%T\n\t%T\n", on->type, n->type);
			on = N;
		}
	}

	// check for forward declaration
	if(on == N) {
		// initial declaration or redeclaration
		// declare fun name, argument types and argument names
		n->nname->type = n->type;
		if(n->type->thistuple == 0)
			addvar(n->nname, n->type, PFUNC);
		else
			n->nname->class = PFUNC;
	} else {
		// identical redeclaration
		// steal previous names
		n->nname = on;
		n->type = on->type;
		n->class = on->class;
		n->sym = s;
	}

	// change the declaration context from extern to auto
	autodcl = list1(nod(OXXX, N, N));

	if(funcdepth == 0 && dclcontext != PEXTERN)
		fatal("funchdr: dclcontext");

	dclcontext = PAUTO;
	markdcl();
	funcargs(n->type);
}

void
funcargs(Type *ft)
{
	Type *t;
	Iter save;
	int all;

	funcdepth++;

	// declare the this/in arguments
	t = funcfirst(&save, ft);
	while(t != T) {
		if(t->nname != N) {
			t->nname->xoffset = t->width;
			addvar(t->nname, t->type, PPARAM);
		}
		t = funcnext(&save);
	}

	// declare the outgoing arguments
	all = 0;
	t = structfirst(&save, getoutarg(ft));
	while(t != T) {
		if(t->nname != N)
			t->nname->xoffset = t->width;
		if(t->nname != N) {
			addvar(t->nname, t->type, PPARAMOUT);
			all |= 1;
		} else
			all |= 2;
		t = structnext(&save);
	}

	// this test is remarkedly similar to checkarglist
	if(all == 3)
		yyerror("cannot mix anonymous and named output arguments");

	ft->outnamed = 0;
	if(all == 1)
		ft->outnamed = 1;
}

/*
 * compile the function.
 * called in auto-declaration context.
 * returns in extern-declaration context.
 */
void
funcbody(Node *n)
{

	compile(n);

	// change the declaration context from auto to extern
	if(dclcontext != PAUTO)
		fatal("funcbody: dclcontext");
	popdcl();
	funcdepth--;
	if(funcdepth == 0)
		dclcontext = PEXTERN;
}

Node*
funclit0(Node *t)
{
	Node *n;

	n = nod(OXXX, N, N);
	n->outer = funclit;
	n->dcl = autodcl;
	funclit = n;

	// new declaration context
	autodcl = list1(nod(OEMPTY, N, N));

	typecheck(&t, Etype);
	funcargs(t->type);
	return t;
}

Node*
funclit1(Node *ntype, NodeList *body)
{
	Node *func;
	Type *type;
	Node *a, *d, *f, *n, *clos;
	Type *ft, *t;
	Iter save;
	int narg, shift;
	NodeList *args, *l, *in, *out;
	static int closgen;

	type = ntype->type;
	popdcl();
	func = funclit;
	funclit = func->outer;

	// build up type of func f that we're going to compile.
	// as we referred to variables from the outer function,
	// we accumulated a list of PHEAP names in func->cvars.
	narg = 0;
	// add PHEAP versions as function arguments.
	in = nil;
	for(l=func->cvars; l; l=l->next) {
		a = l->n;
		d = nod(ODCLFIELD, a, N);
		d->type = ptrto(a->type);
		in = list(in, d);

		// while we're here, set up a->heapaddr for back end
		n = nod(ONAME, N, N);
		snprint(namebuf, sizeof namebuf, "&%s", a->sym->name);
		n->sym = lookup(namebuf);
		n->type = ptrto(a->type);
		n->class = PPARAM;
		n->xoffset = narg*types[tptr]->width;
		n->addable = 1;
		n->ullman = 1;
		narg++;
		a->heapaddr = n;

		a->xoffset = 0;

		// unlink from actual ONAME in symbol table
		a->closure->closure = a->outer;
	}

	// add a dummy arg for the closure's caller pc
	d = nod(ODCLFIELD, N, N);
	d->type = types[TUINTPTR];
	in = list(in, d);

	// slide param offset to make room for ptrs above.
	// narg+1 to skip over caller pc.
	shift = (narg+1)*types[tptr]->width;

	// now the original arguments.
	for(t=structfirst(&save, getinarg(type)); t; t=structnext(&save)) {
		d = nod(ODCLFIELD, t->nname, N);
		d->type = t->type;
		in = list(in, d);

		a = t->nname;
		if(a != N) {
			if(a->stackparam != N)
				a = a->stackparam;
			a->xoffset += shift;
		}
	}

	// out arguments
	out = nil;
	for(t=structfirst(&save, getoutarg(type)); t; t=structnext(&save)) {
		d = nod(ODCLFIELD, t->nname, N);
		d->type = t->type;
		out = list(out, d);

		a = t->nname;
		if(a != N) {
			if(a->stackparam != N)
				a = a->stackparam;
			a->xoffset += shift;
		}
	}

	ft = functype(N, in, out);
	ft->outnamed = type->outnamed;

	// declare function.
	snprint(namebuf, sizeof(namebuf), "_f%.3ld·%s", ++closgen, filename);
	f = newname(lookup(namebuf));
	addvar(f, ft, PFUNC);
	f->funcdepth = 0;

	// compile function
	n = nod(ODCLFUNC, N, N);
	n->nname = f;
	n->type = ft;
	if(body == nil)
		body = list1(nod(OEMPTY, N, N));
	n->nbody = body;
	compile(n);
	funcdepth--;
	autodcl = func->dcl;

	// build up type for this instance of the closure func.
	in = nil;
	d = nod(ODCLFIELD, N, N);	// siz
	d->type = types[TINT];
	in = list(in, d);
	d = nod(ODCLFIELD, N, N);	// f
	d->type = ft;
	in = list(in, d);
	for(l=func->cvars; l; l=l->next) {
		a = l->n;
		d = nod(ODCLFIELD, N, N);	// arg
		d->type = ptrto(a->type);
		in = list(in, d);
	}

	d = nod(ODCLFIELD, N, N);
	d->type = type;
	out = list1(d);

	clos = syslook("closure", 1);
	clos->type = functype(N, in, out);

	// literal expression is sys.closure(siz, f, arg0, arg1, ...)
	// which builds a function that calls f after filling in arg0,
	// arg1, ... for the PHEAP arguments above.
	args = nil;
	if(narg*widthptr > 100)
		yyerror("closure needs too many variables; runtime will reject it");
	a = nodintconst(narg*widthptr);
	args = list(args, a);	// siz
	args = list(args, f);	// f
	for(l=func->cvars; l; l=l->next) {
		a = l->n;
		d = oldname(a->sym);
		args = list(args, nod(OADDR, d, N));
	}
	typechecklist(args, Erv);

	n = nod(OCALL, clos, N);
	n->list = args;
	return n;
}
