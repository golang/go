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

void
dodclvar(Node *n, Type *t)
{
	if(n == N)
		return;

	for(; n->op == OLIST; n = n->right)
		dodclvar(n->left, t);

	dowidth(t);

	// in case of type checking error,
	// use "undefined" type for variable type,
	// to avoid fatal in addvar.
	if(t == T)
		t = typ(TFORW);

	addvar(n, t, dclcontext);
	if(dcladj)
		dcladj(n->sym);
}

void
dodclconst(Node *n, Node *e)
{
	if(n == N)
		return;

	for(; n->op == OLIST; n=n->right)
		dodclconst(n, e);

	addconst(n, e, dclcontext);
	if(dcladj)
		dcladj(n->sym);
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
	if(s->block == block) {
		switch(s->otype->etype) {
		case TFORWSTRUCT:
		case TFORWINTER:
			n = s->otype;
			goto found;
		}
	}

	// otherwise declare a new type
	addtyp(n, dclcontext);

found:
	n->local = 1;
	if(dcladj)
		dcladj(n->sym);
	return n;
}

/*
 * now we know what n is: it's t
 */
void
updatetype(Type *n, Type *t)
{
	Sym *s;

	s = n->sym;
	if(s == S || s->otype != n)
		fatal("updatetype %T = %T", n, t);

	switch(n->etype) {
	case TFORW:
		break;

	case TFORWSTRUCT:
		if(t->etype != TSTRUCT) {
			yyerror("%T forward declared as struct", n);
			return;
		}
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

	if(n->local)
		t->local = 1;
	*n = *t;
	n->sym = s;

	// catch declaration of incomplete type
	switch(n->etype) {
	case TFORWSTRUCT:
	case TFORWINTER:
		break;
	default:
		checkwidth(n);
	}
}


/*
 * return nelem of list
 */
int
listcount(Node *n)
{
	int v;

	v = 0;
	while(n != N) {
		v++;
		if(n->op != OLIST)
			break;
		n = n->right;
	}
	return v;
}

/*
 * turn a parsed function declaration
 * into a type
 */
Type*
functype(Node *this, Node *in, Node *out)
{
	Type *t;

	t = typ(TFUNC);

	t->type = dostruct(this, TFUNC);
	t->type->down = dostruct(out, TFUNC);
	t->type->down->down = dostruct(in, TFUNC);

	t->thistuple = listcount(this);
	t->outtuple = listcount(out);
	t->intuple = listcount(in);

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

		if(!eqtype(t1->type, t2->type, 0))
			return 0;

		t1 = t1->down;
		t2 = t2->down;
	}
	return 1;
}

Sym*
methodsym(Sym *nsym, Type *t)
{
	Sym *s;
	char buf[NSYMB];

	// caller has already called ismethod to obtain t
	if(t == T)
		goto bad;
	s = t->sym;
	if(s == S)
		goto bad;

	snprint(buf, sizeof(buf), "%#hT·%s", t, nsym->name);
//print("methodname %s\n", buf);
	return pkglookup(buf, s->opackage);

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
	Sym *st, *sf;

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

	// and finally the receiver sym
	f = ismethod(pa);
	if(f == T)
		goto bad;
	pa = f;
	st = pa->sym;
	if(st == S)
		goto bad;
	if(local && !f->local) {
		yyerror("method receiver type must be locally defined: %T", f);
		return;
	}

	n = nod(ODCLFIELD, newname(sf), N);
	n->type = t;

	d = T;	// last found
	for(f=pa->method; f!=T; f=f->down) {
		if(f->etype != TFIELD)
			fatal("addmethod: not TFIELD: %N", f);

		if(strcmp(sf->name, f->sym->name) != 0) {
			d = f;
			continue;
		}
		if(!eqtype(t, f->type, 0)) {
			yyerror("method redeclared: %S of type %S", sf, st);
			print("\t%T\n\t%T\n", f->type, t);
		}
		return;
	}

	if(d == T)
		stotype(n, &pa->method);
	else
		stotype(n, &d->down);

	if(dflag())
		print("method         %S of type %T\n", sf, pa);
	return;

bad:
	yyerror("unknown method pointer: %T %S", pa, sf);
}

/*
 * a function named init is a special case.
 * it is called by the initialization before
 * main is run. to make it unique within a
 * package, the name, normally "pkg.init", is
 * altered to "pkg.<file>_init".
 */
Node*
renameinit(Node *n)
{
	Sym *s;

	s = n->sym;
	if(s == S)
		return n;
	if(strcmp(s->name, "init") != 0)
		return n;
	snprint(namebuf, sizeof(namebuf), "init_%s", filename);
	s = lookup(namebuf);
	return newname(s);
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
	on = s->oname;

	// check for same types
	if(on != N) {
		if(eqtype(n->type, on->type, 0)) {
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
			addvar(n->nname, n->type, PEXTERN);
		else
			n->nname->class = PEXTERN;
	} else {
		// identical redeclaration
		// steal previous names
		n->nname = on;
		n->type = on->type;
		n->class = on->class;
		n->sym = s;
		if(dflag())
			print("forew  var-dcl %S %T\n", n->sym, n->type);
	}

	// change the declaration context from extern to auto
	autodcl = dcl();
	autodcl->back = autodcl;

	if(dclcontext != PEXTERN)
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
		if(t->nname != N && t->nname->sym->name[0] != '_') {
			addvar(t->nname, t->type, PPARAM);
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
	dclcontext = PEXTERN;
}

/*
 * turn a parsed struct into a type
 */
Type**
stotype(Node *n, Type **t)
{
	Type *f;
	Iter save;
	String *note;

	n = listfirst(&save, &n);

loop:
	note = nil;
	if(n == N) {
		*t = T;
		return t;
	}

	if(n->op == OLIST) {
		// recursive because it can be lists of lists
		t = stotype(n, t);
		goto next;
	}

	if(n->op != ODCLFIELD || n->type == T)
		fatal("stotype: oops %N\n", n);

	switch(n->type->etype) {
	case TCHAN:
	case TMAP:
	case TSTRING:
		yyerror("%T can exist only in pointer form", n->type);
		break;
	}

	switch(n->val.ctype) {
	case CTSTR:
		note = n->val.u.sval;
		break;
	default:
		yyerror("structure field annotation must be string");
	case CTxxx:
		note = nil;
		break;
	}

	f = typ(TFIELD);
	f->type = n->type;
	f->note = note;
	f->width = BADWIDTH;

	if(n->left != N && n->left->op == ONAME) {
		f->nname = n->left;
		f->embedded = n->embedded;
		f->sym = f->nname->sym;
	}

	*t = f;
	t = &f->down;

next:
	n = listnext(&save);
	goto loop;
}

Type*
dostruct(Node *n, int et)
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
	stotype(n, &t->type);
	if(!funarg)
		checkwidth(t);
	return t;
}

Type*
sortinter(Type *t)
{
	return t;
}

void
dcopy(Sym *a, Sym *b)
{
	a->name = b->name;
	a->oname = b->oname;
	a->otype = b->otype;
	a->oconst = b->oconst;
	a->package = b->package;
	a->opackage = b->opackage;
	a->lexical = b->lexical;
	a->undef = b->undef;
	a->vargen = b->vargen;
	a->block = b->block;
	a->lastlineno = b->lastlineno;
	a->offset = b->offset;
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
markdclstack(void)
{
	Sym *d, *s;

	markdcl();

	// copy the entire pop of the stack
	// all the way back to block0.
	// after this the symbol table is at
	// block0 and popdcl will restore it.
	for(d=dclstack; d!=S; d=d->link) {
		if(d == b0stack)
			break;
		if(d->name != nil) {
			s = pkglookup(d->name, d->package);
			pushdcl(s);
			dcopy(s, d);
		}
	}
}

void
dumpdcl(char *st)
{
	Sym *s, *d;
	int i;

	print("\ndumpdcl: %s %p\n", st, b0stack);

	i = 0;
	for(d=dclstack; d!=S; d=d->link) {
		i++;
		print("    %.2d %p", i, d);
		if(d == b0stack)
			print(" (b0)");
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

static void
redeclare(char *str, Sym *s)
{
	if(s->block == block) {
		yyerror("%s %S redeclared in this block", str, s);
		print("	previous declaration at %L\n", s->lastlineno);
	}
	s->block = block;
	s->lastlineno = lineno;
}

void
addvar(Node *n, Type *t, int ctxt)
{
	Dcl *r, *d;
	Sym *s;
	int gen;

	if(n==N || n->sym == S || n->op != ONAME || t == T)
		fatal("addvar: n=%N t=%T nil", n, t);

	s = n->sym;

	if(ctxt == PEXTERN) {
		r = externdcl;
		gen = 0;
	} else {
		r = autodcl;
		vargen++;
		gen = vargen;
		pushdcl(s);
	}

	if(t != T) {
		switch(t->etype) {
		case TCHAN:
		case TMAP:
		case TSTRING:
			yyerror("%T can exist only in pointer form", t);
		}
	}

	redeclare("variable", s);
	s->vargen = gen;
	s->oname = n;
	s->offset = 0;
	s->lexical = LNAME;

	n->type = t;
	n->vargen = gen;
	n->class = ctxt;

	d = dcl();
	d->dsym = s;
	d->dnode = n;
	d->op = ONAME;

	r->back->forw = d;
	r->back = d;

	if(dflag()) {
		if(ctxt == PEXTERN)
			print("extern var-dcl %S G%ld %T\n", s, s->vargen, t);
		else
			print("auto   var-dcl %S G%ld %T\n", s, s->vargen, t);
	}
}

void
addtyp(Type *n, int ctxt)
{
	Dcl *r, *d;
	Sym *s;
	static int typgen;

	if(n==T || n->sym == S)
		fatal("addtyp: n=%T t=%T nil", n);

	s = n->sym;

	if(ctxt == PEXTERN)
		r = externdcl;
	else {
		r = autodcl;
		pushdcl(s);
		n->vargen = ++typgen;
	}

	redeclare("type", s);
	s->otype = n;
	s->lexical = LATYPE;

	d = dcl();
	d->dsym = s;
	d->dtype = n;
	d->op = OTYPE;

	d->back = r->back;
	r->back->forw = d;
	r->back = d;

	d = dcl();
	d->dtype = n;
	d->op = OTYPE;

	r = typelist;
	d->back = r->back;
	r->back->forw = d;
	r->back = d;

	if(dflag()) {
		if(ctxt == PEXTERN)
			print("extern typ-dcl %S G%ld %T\n", s, s->vargen, n);
		else
			print("auto   typ-dcl %S G%ld %T\n", s, s->vargen, n);
	}
}

void
addconst(Node *n, Node *e, int ctxt)
{
	Sym *s;
	Dcl *r, *d;

	if(n->op != ONAME)
		fatal("addconst: not a name");

	if(e->op != OLITERAL) {
		yyerror("expression must be a constant");
		return;
	}

	s = n->sym;

	if(ctxt == PEXTERN)
		r = externdcl;
	else {
		r = autodcl;
		pushdcl(s);
	}

	redeclare("constant", s);
	s->oconst = e;
	s->lexical = LACONST;

	d = dcl();
	d->dsym = s;
	d->dnode = e;
	d->op = OCONST;
	d->back = r->back;
	r->back->forw = d;
	r->back = d;

	if(dflag())
		print("const-dcl %S %N\n", n->sym, n->sym->oconst);
}

Node*
fakethis(void)
{
	Node *n;
	Type *t;

	n = nod(ODCLFIELD, N, N);
	t = dostruct(N, TSTRUCT);
	t = ptrto(t);
	n->type = t;
	return n;
}

/*
 * this generates a new name that is
 * pushed down on the declaration list.
 * no diagnostics are produced as this
 * name will soon be declared.
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
	return n;
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

	n = s->oname;
	if(n == N) {
		n = nod(ONONAME, N, N);
		n->sym = s;
		n->type = T;
		n->addable = 1;
		n->ullman = 1;
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

Type*
oldtype(Sym *s)
{
	Type *t;

	t = s->otype;
	if(t == T)
		fatal("%S not a type", s); // cant happen
	return t;
}

/*
 * n is a node with a name (or a reversed list of them).
 * make it an anonymous declaration of that name's type.
 */
Node*
nametoanondcl(Node *na)
{
	Node **l, *n;
	Type *t;

	for(l=&na; (n=*l)->op == OLIST; l=&n->left)
		n->right = nametoanondcl(n->right);

	if(n->sym->lexical != LATYPE && n->sym->lexical != LBASETYPE) {
		yyerror("%s is not a type", n->sym->name);
		t = typ(TINT32);
	} else
		t = oldtype(n->sym);
	n = nod(ODCLFIELD, N, N);
	n->type = t;
	*l = n;
	return na;
}

/*
 * n is a node with a name (or a reversed list of them).
 * make it a declaration of the given type.
 */
Node*
nametodcl(Node *na, Type *t)
{
	Node **l, *n;

	for(l=&na; (n=*l)->op == OLIST; l=&n->left)
		n->right = nametodcl(n->right, t);

	n = nod(ODCLFIELD, n, N);
	n->type = t;
	*l = n;
	return na;
}

/*
 * make an anonymous declaration for t
 */
Node*
anondcl(Type *t)
{
	Node *n;

	n = nod(ODCLFIELD, N, N);
	n->type = t;
	return n;
}

/*
 * check that the list of declarations is either all anonymous or all named
 */
void
checkarglist(Node *n)
{
	if(n->op != OLIST)
		return;
	if(n->left->op != ODCLFIELD)
		fatal("checkarglist");
	if(n->left->left != N) {
		for(n=n->right; n->op == OLIST; n=n->right)
			if(n->left->left == N)
				goto mixed;
		if(n->left == N)
			goto mixed;
	} else {
		for(n=n->right; n->op == OLIST; n=n->right)
			if(n->left->left != N)
				goto mixed;
		if(n->left != N)
			goto mixed;
	}
	return;

mixed:
	yyerror("cannot mix anonymous and named function arguments");
}

// hand-craft the following initialization code
//	var	init_<file>_done bool;			(1)
//	func	init_<file>_function()			(2)
//		if init_<file>_done { return }		(3)
//		init_<file>_done = true;		(4)
//		// over all matching imported symbols
//			<pkg>.init_<file>_function()	(5)
//		{ <init stmts> }			(6)
//		init()	// if any			(7)
//		return					(8)
//	}
//	export	init_<file>_function			(9)

void
fninit(Node *n)
{
	Node *done;
	Node *a, *fn, *r;
	uint32 h;
	Sym *s;

	r = N;

	// (1)
	snprint(namebuf, sizeof(namebuf), "init_%s_done", filename);
	done = newname(lookup(namebuf));
	addvar(done, types[TBOOL], PEXTERN);

	// (2)

	maxarg = 0;
	stksize = initstksize;

	snprint(namebuf, sizeof(namebuf), "init_%s_function", filename);

	// this is a botch since we need a known name to
	// call the top level init function out of rt0
	if(strcmp(package, "main") == 0)
		snprint(namebuf, sizeof(namebuf), "init_function");

	fn = nod(ODCLFUNC, N, N);
	fn->nname = newname(lookup(namebuf));
	fn->type = functype(N, N, N);
	funchdr(fn);

	// (3)
	a = nod(OIF, N, N);
	a->ntest = done;
	a->nbody = nod(ORETURN, N, N);
	r = list(r, a);

	// (4)
	a = nod(OAS, done, booltrue);
	r = list(r, a);

	// (5)
	for(h=0; h<NHASH; h++)
	for(s = hash[h]; s != S; s = s->link) {
		if(s->name[0] != 'i')
			continue;
		if(strstr(s->name, "init_") == nil)
			continue;
		if(strstr(s->name, "_function") == nil)
			continue;
		if(s->oname == N)
			continue;

		// could check that it is fn of no args/returns
		a = nod(OCALL, s->oname, N);
		r = list(r, a);
	}

	// (6)
	r = list(r, n);

	// (7)
	// could check that it is fn of no args/returns
	snprint(namebuf, sizeof(namebuf), "init_%s", filename);
	s = lookup(namebuf);
	if(s->oname != N) {
		a = nod(OCALL, s->oname, N);
		r = list(r, a);
	}

	// (8)
	a = nod(ORETURN, N, N);
	r = list(r, a);

	// (9)
	exportsym(fn->nname->sym);

	fn->nbody = rev(r);
//dump("b", fn);
//dump("r", fn->nbody);

	popdcl();
	compile(fn);
}


/*
 * when a type's width should be known, we call checkwidth
 * to compute it.  during a declaration like
 *
 *	type T *struct { next T }
 *
 * it is necessary to defer the calculation of the struct width
 * until after T has been initialized to be a pointer to that struct.
 * similarly, during import processing structs may be used
 * before their definition.  in those situations, calling
 * defercheckwidth() stops width calculations until
 * resumecheckwidth() is called, at which point all the
 * checkwidths that were deferred are executed.
 * sometimes it is okay to
 */
typedef struct TypeList TypeList;
struct TypeList {
	Type *t;
	TypeList *next;
};

static TypeList *tlfree;
static TypeList *tlq;
static int defercalc;

void
checkwidth(Type *t)
{
	TypeList *l;

	// function arg structs should not be checked
	// outside of the enclosing function.
	if(t->funarg)
		fatal("checkwidth %T", t);

	if(!defercalc) {
		dowidth(t);
		return;
	}

	l = tlfree;
	if(l != nil)
		tlfree = l->next;
	else
		l = mal(sizeof *l);

	l->t = t;
	l->next = tlq;
	tlq = l;
}

void
defercheckwidth(void)
{
	// we get out of sync on syntax errors, so don't be pedantic.
	// if(defercalc)
	//	fatal("defercheckwidth");
	defercalc = 1;
}

void
resumecheckwidth(void)
{
	TypeList *l;

	if(!defercalc)
		fatal("restartcheckwidth");
	defercalc = 0;

	for(l = tlq; l != nil; l = tlq) {
		dowidth(l->t);
		tlq = l->next;
		l->next = tlfree;
		tlfree = l;
	}
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
	n->type = oldtype(s);
	if(isptr[n->type->etype])
		yyerror("embedded type cannot be a pointer");
	return n;
}

/*
 * declare variables from grammar
 * new_name_list [type] = expr_list
 */
Node*
variter(Node *vv, Type *t, Node *ee)
{
	Iter viter, eiter;
	Node *v, *e, *r, *a;

	vv = rev(vv);
	ee = rev(ee);

	v = listfirst(&viter, &vv);
	e = listfirst(&eiter, &ee);
	r = N;

loop:
	if(v == N && e == N)
		return rev(r);

	if(v == N || e == N) {
		yyerror("shape error in var dcl");
		return rev(r);
	}

	a = nod(OAS, v, N);
	if(t == T) {
		gettype(e, a);
		defaultlit(e);
		dodclvar(v, e->type);
	} else
		dodclvar(v, t);
	a->right = e;

	r = list(r, a);

	v = listnext(&viter);
	e = listnext(&eiter);
	goto loop;
}

/*
 * declare constants from grammar
 * new_name_list [type] [= expr_list]
 */
void
constiter(Node *vv, Type *t, Node *cc)
{
	Iter viter, citer;
	Node *v, *c;

	if(cc == N)
		cc = lastconst;
	lastconst = cc;
	vv = rev(vv);
	cc = rev(treecopy(cc));

	v = listfirst(&viter, &vv);
	c = listfirst(&citer, &cc);

loop:
	if(v == N && c == N) {
		iota += 1;
		return;
	}

	if(v == N || c == N) {
		yyerror("shape error in var dcl");
		iota += 1;
		return;
	}

	gettype(c, N);
	if(t != T)
		convlit(c, t);
	dodclconst(v, c);

	v = listnext(&viter);
	c = listnext(&citer);
	goto loop;
}
