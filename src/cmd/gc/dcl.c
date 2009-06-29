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

	if(t != T && (t->etype == TIDEAL || t->etype == TNIL))
		fatal("dodclvar %T", t);
	for(; n->op == OLIST; n = n->right)
		dodclvar(n->left, t);

	dowidth(t);

	// in case of type checking error,
	// use "undefined" type for variable type,
	// to avoid fatal in addvar.
	if(t == T)
		t = typ(TFORW);

	addvar(n, t, dclcontext);
	autoexport(n->sym);
	if(funcdepth > 0)
		addtop = list(addtop, nod(ODCL, n, N));
}

void
dodclconst(Node *n, Node *e)
{
	if(n == N)
		return;

	for(; n->op == OLIST; n=n->right)
		dodclconst(n, e);

	addconst(n, e, dclcontext);
	autoexport(n->sym);
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
	if(s->block == block && s->def != N && s->def->op == OTYPE) {
		switch(s->def->type->etype) {
		case TFORWSTRUCT:
		case TFORWINTER:
			n = s->def->type;
			goto found;
		}
	}

	// otherwise declare a new type
	addtyp(n, dclcontext);

found:
	n->local = 1;
	autoexport(n->sym);
	return n;
}

/*
 * now we know what n is: it's t
 */
void
updatetype(Type *n, Type *t)
{
	Sym *s;
	int local;
	int maplineno, lno, etype;

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
	*n = *t;
	n->sym = s;
	n->local = local;
	n->siggen = 0;
	n->printed = 0;
	n->method = nil;
	n->vargen = 0;
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
 * return nelem of list
 */
int
listcount(Node *n)
{
	int v;
	Iter s;

	v = 0;
	for(n = listfirst(&s, &n); n != N; n = listnext(&s))
		v++;
	return v;
}

int
structcount(Type *t)
{
	int v;
	Iter s;

	v = 0;
	for(t = structfirst(&s, &t); t != T; t = structnext(&s))
		v++;
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
		stotype(n, 0, &pa->method);
	else
		stotype(n, 0, &d->down);

	if(dflag())
		print("method         %S of type %T\n", sf, pa);
	return;

bad:
	yyerror("invalid receiver type %T", pa);
}

/*
 * a function named init is a special case.
 * it is called by the initialization before
 * main is run. to make it unique within a
 * package and also uncallable, the name,
 * normally "pkg.init", is altered to "pkg.init·filename".
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

	snprint(namebuf, sizeof(namebuf), "init·%s", filename);
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
		if(dflag())
			print("forew  var-dcl %S %T\n", n->sym, n->type);
	}

	// change the declaration context from extern to auto
	autodcl = dcl();
	autodcl->back = autodcl;

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

void
funclit0(Type *t)
{
	Node *n;

	n = nod(OXXX, N, N);
	n->outer = funclit;
	n->dcl = autodcl;
	funclit = n;

	// new declaration context
	autodcl = dcl();
	autodcl->back = autodcl;

	funcargs(t);
}

Node*
funclit1(Type *type, Node *body)
{
	Node *func;
	Node *a, *d, *f, *n, *args, *clos, *in, *out;
	Type *ft, *t;
	Iter save;
	int narg, shift;

	popdcl();
	func = funclit;
	funclit = func->outer;

	// build up type of func f that we're going to compile.
	// as we referred to variables from the outer function,
	// we accumulated a list of PHEAP names in func.
	//
	narg = 0;
	if(func->cvars == N)
		ft = type;
	else {
		// add PHEAP versions as function arguments.
		in = N;
		for(a=listfirst(&save, &func->cvars); a; a=listnext(&save)) {
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
		d = nod(ODCLFIELD, a, N);
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
		in = rev(in);

		// out arguments
		out = N;
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
		out = rev(out);

		ft = functype(N, in, out);
		ft->outnamed = type->outnamed;
	}

	// declare function.
	vargen++;
	snprint(namebuf, sizeof(namebuf), "_f%.3ld·%s", vargen, filename);
	f = newname(lookup(namebuf));
	addvar(f, ft, PFUNC);
	f->funcdepth = 0;

	// compile function
	n = nod(ODCLFUNC, N, N);
	n->nname = f;
	n->type = ft;
	if(body == N)
		body = nod(ORETURN, N, N);
	n->nbody = body;
	compile(n);
	funcdepth--;
	autodcl = func->dcl;

	// if there's no closure, we can use f directly
	if(func->cvars == N)
		return f;

	// build up type for this instance of the closure func.
	in = N;
	d = nod(ODCLFIELD, N, N);	// siz
	d->type = types[TINT];
	in = list(in, d);
	d = nod(ODCLFIELD, N, N);	// f
	d->type = ft;
	in = list(in, d);
	for(a=listfirst(&save, &func->cvars); a; a=listnext(&save)) {
		d = nod(ODCLFIELD, N, N);	// arg
		d->type = ptrto(a->type);
		in = list(in, d);
	}
	in = rev(in);

	d = nod(ODCLFIELD, N, N);
	d->type = type;
	out = d;

	clos = syslook("closure", 1);
	clos->type = functype(N, in, out);

	// literal expression is sys.closure(siz, f, arg0, arg1, ...)
	// which builds a function that calls f after filling in arg0,
	// arg1, ... for the PHEAP arguments above.
	args = N;
	if(narg*widthptr > 100)
		yyerror("closure needs too many variables; runtime will reject it");
	a = nodintconst(narg*widthptr);
	args = list(args, a);	// siz
	args = list(args, f);	// f
	for(a=listfirst(&save, &func->cvars); a; a=listnext(&save)) {
		d = oldname(a->sym);
		addrescapes(d);
		args = list(args, nod(OADDR, d, N));
	}
	args = rev(args);

	return nod(OCALL, clos, args);
}



/*
 * turn a parsed struct into a type
 */
Type**
stotype(Node *n, int et, Type **t)
{
	Type *f, *t1;
	Iter save;
	Strlit *note;
	int lno;

	lno = lineno;
	n = listfirst(&save, &n);

loop:
	note = nil;
	if(n == N) {
		*t = T;
		lineno = lno;
		return t;
	}

	lineno = n->lineno;
	if(n->op == OLIST) {
		// recursive because it can be lists of lists
		t = stotype(n, et, t);
		goto next;
	}

	if(n->op != ODCLFIELD)
		fatal("stotype: oops %N\n", n);

	if(n->type == T) {
		// assume error already printed
		goto next;
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
			goto next;
		}
		for(t1=n->type->type; t1!=T; t1=t1->down) {
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
		goto next;
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
	stotype(n, et, &t->type);
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
	a->def = b->def;
	a->package = b->package;
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

	if(ctxt == PEXTERN || ctxt == PFUNC) {
		r = externdcl;
		gen = 0;
	} else {
		r = autodcl;
		vargen++;
		gen = vargen;
		pushdcl(s);
	}

	redeclare("variable", s);
	s->vargen = gen;
	s->def = n;
	s->offset = 0;

	n->funcdepth = funcdepth;
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
		else if(ctxt == PFUNC)
			print("extern func-dcl %S G%ld %T\n", s, s->vargen, t);
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
	s->def = typenod(n);

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

	if(n->op != ONAME && n->op != ONONAME)
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
	s->def = e;
	e->sym = s;

	d = dcl();
	d->dsym = s;
	d->dnode = e;
	d->op = OLITERAL;
	d->back = r->back;
	r->back->forw = d;
	r->back = d;

	if(dflag())
		print("const-dcl %S %N\n", n->sym, n->sym->def);
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
	n->xoffset = 0;
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
		n = nod(ONONAME, N, N);
		n->sym = s;
		n->type = T;
		n->addable = 1;
		n->ullman = 1;
	}
	if(n->op == OLITERAL) {
		c = nod(OLITERAL, N, N);
		c->sym = s;
		c->val = n->val;
		c->type = n->type;
		c->iota = n->iota;
		return c;
	}
	if(n->funcdepth > 0 && n->funcdepth != funcdepth && n->op == ONAME) {
		// inner func is referring to var
		// in outer func.
		if(n->closure == N || n->closure->funcdepth != funcdepth) {
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
			funclit->cvars = list(c, funclit->cvars);
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

Type*
oldtype(Sym *s)
{
	Type *t;

	if(s == S)
		return T;
	if(s->def == N || s->def->op != OTYPE) {
		yyerror("%S is not a type", s);
		return T;
	}
	t = s->def->type;

	/*
	 * If t is lowercase and not in our package
	 * and this isn't a reference during the parsing
	 * of import data, complain.
	 */
	if(pkgimportname == S && !exportname(s->name) && strcmp(s->package, package) != 0)
		yyerror("cannot use type %T", t);
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

	n = n->sym->def;
	if(n == N || n->op != OTYPE || (t = n->type) == T) {
		yyerror("%S is not a type", n->sym);
		t = typ(TINT32);
	}
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

static Node*
findtype(Node *n)
{
	Node *r;

	for(r=n; r->op==OLIST; r=r->right)
		if(r->left->op == OKEY)
			return r->left->right;
	if(r->op == OKEY)
		return r->right;
	if(n->op == OLIST)
		n = n->left;
	return N;
}

static Node*
xanondcl(Node *nt, int dddok)
{
	Node *n;
	Type *t;

	t = nt->type;
	if(nt->op != OTYPE) {
		yyerror("%S is not a type", nt->sym);
		t = types[TINT32];
	}
	n = nod(ODCLFIELD, N, N);
	n->type = t;
	if(!dddok && t->etype == TDDD)
		yyerror("only last argument can have type ...");
	return n;
}

static Node*
namedcl(Node *nn, Node *nt, int dddok)
{
	Node *n;
	Type *t;

	if(nn->op == OKEY)
		nn = nn->left;
	if(nn->op == OTYPE && nn->sym == S) {
		yyerror("cannot mix anonymous %T with named arguments", nn->type);
		return xanondcl(nn, dddok);
	}
	t = types[TINT32];
	if(nt == N)
		yyerror("missing type for argument %S", nn->sym);
	else if(nt->op != OTYPE)
		yyerror("%S is not a type", nt->sym);
	else
		t = nt->type;
	n = nod(ODCLFIELD, newname(nn->sym), N);
	n->type = t;
	if(!dddok && t->etype == TDDD)
		yyerror("only last argument can have type ...");
	return n;
}

/*
 * check that the list of declarations is either all anonymous or all named
 */
Node*
checkarglist(Node *n)
{
	Node *r;
	Node **l;

	// check for all anonymous
	for(r=n; r->op==OLIST; r=r->right)
		if(r->left->op == OKEY)
			goto named;
	if(r->op == OKEY)
		goto named;

	// all anonymous - add names
	for(l=&n; (r=*l)->op==OLIST; l=&r->right)
		r->left = xanondcl(r->left, 0);
	*l = xanondcl(r, 1);
	return n;


named:
	// otherwise, each run of names ends in a type.
	// add a type to each one that needs one.
	for(l=&n; (r=*l)->op==OLIST; l=&r->right)
		r->left = namedcl(r->left, findtype(r), 0);
	*l = namedcl(r, findtype(r), 1);
	return n;
}

/*
 * hand-craft the following initialization code
 *	var initdone·<file> uint8 			(1)
 *	func	Init·<file>()				(2)
 *		if initdone·<file> {			(3)
 *			if initdone·<file> == 2		(4)
 *				return
 *			throw();			(5)
 *		}
 *		initdone.<file>++;			(6)
 *		// over all matching imported symbols
 *			<pkg>.init·<file>()		(7)
 *		{ <init stmts> }			(8)
 *		init·<file>()	// if any		(9)
 *		initdone.<file>++;			(10)
 *		return					(11)
 *	}
 */
int
anyinit(Node *n)
{
	uint32 h;
	Sym *s;

	// are there any init statements
	if(n != N)
		return 1;

	// is this main
	if(strcmp(package, "main") == 0)
		return 1;

	// is there an explicit init function
	snprint(namebuf, sizeof(namebuf), "init·%s", filename);
	s = lookup(namebuf);
	if(s->def != N)
		return 1;

	// are there any imported init functions
	for(h=0; h<NHASH; h++)
	for(s = hash[h]; s != S; s = s->link) {
		if(s->name[0] != 'I' || strncmp(s->name, "Init·", 6) != 0)
			continue;
		if(s->def == N)
			continue;
		return 1;
	}

	// then none
	return 0;
}

void
fninit(Node *n)
{
	Node *gatevar;
	Node *a, *b, *fn, *r;
	uint32 h;
	Sym *s, *initsym;

	if(strcmp(package, "PACKAGE") == 0) {
		// sys.go or unsafe.go during compiler build
		return;
	}

	if(!anyinit(n))
		return;

	r = N;

	// (1)
	snprint(namebuf, sizeof(namebuf), "initdone·%s", filename);
	gatevar = newname(lookup(namebuf));
	addvar(gatevar, types[TUINT8], PEXTERN);

	// (2)

	maxarg = 0;
	stksize = initstksize;

	snprint(namebuf, sizeof(namebuf), "Init·%s", filename);

	// this is a botch since we need a known name to
	// call the top level init function out of rt0
	if(strcmp(package, "main") == 0)
		snprint(namebuf, sizeof(namebuf), "init");

	fn = nod(ODCLFUNC, N, N);
	initsym = lookup(namebuf);
	fn->nname = newname(initsym);
	fn->type = functype(N, N, N);
	funchdr(fn);

	// (3)
	a = nod(OIF, N, N);
	a->ntest = nod(ONE, gatevar, nodintconst(0));
	r = list(r, a);

	// (4)
	b = nod(OIF, N, N);
	b->ntest = nod(OEQ, gatevar, nodintconst(2));
	b->nbody = nod(ORETURN, N, N);
	a->nbody = b;

	// (5)
	b = syslook("throwinit", 0);
	b = nod(OCALL, b, N);
	a->nbody = list(a->nbody, b);

	// (6)
	a = nod(OASOP, gatevar, nodintconst(1));
	a->etype = OADD;
	r = list(r, a);

	// (7)
	for(h=0; h<NHASH; h++)
	for(s = hash[h]; s != S; s = s->link) {
		if(s->name[0] != 'I' || strncmp(s->name, "Init·", 6) != 0)
			continue;
		if(s->def == N)
			continue;
		if(s == initsym)
			continue;

		// could check that it is fn of no args/returns
		a = nod(OCALL, s->def, N);
		r = list(r, a);
	}

	// (8)
	r = list(r, initfix(n));

	// (9)
	// could check that it is fn of no args/returns
	snprint(namebuf, sizeof(namebuf), "init·%s", filename);
	s = lookup(namebuf);
	if(s->def != N) {
		a = nod(OCALL, s->def, N);
		r = list(r, a);
	}

	// (10)
	a = nod(OASOP, gatevar, nodintconst(1));
	a->etype = OADD;
	r = list(r, a);

	// (11)
	a = nod(ORETURN, N, N);
	r = list(r, a);

	exportsym(fn->nname->sym);

	fn->nbody = rev(r);
//dump("b", fn);
//dump("r", fn->nbody);

	popdcl();
	initflag = 1;	// flag for loader static initialization
	compile(fn);
	initflag = 0;
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
	if(n->type != T && isptr[n->type->etype])
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
		defaultlit(e, T);
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
 * new_name_list [[type] = expr_list]
 */
void
constiter(Node *vv, Type *t, Node *cc)
{
	Iter viter, citer;
	Node *v, *c, n1;

	if(cc == N) {
		if(t != T)
			yyerror("constdcl cannot have type without expr");
		cc = lastconst;
		t = lasttype;
	}
	lastconst = cc;
	lasttype = t;
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
		yyerror("shape error in const dcl");
		iota += 1;
		return;
	}

	memset(&n1, 0, sizeof n1);
	gettype(c, &n1);
	if(n1.ninit != nil) {
		// the expression had extra code to run.
		// dodclconst is going to print an error
		// because the expression isn't constant,
		// but out of paranoia, bump nerrors so
		// that compile cannot succeed accidentally
		nerrors++;
	}
	if(t != T)
		convlit(c, t);
	if(t == T)
		lasttype = c->type;
	dodclconst(v, c);

	v = listnext(&viter);
	c = listnext(&citer);
	goto loop;
}

/*
 * look for
 *	unsafe.Sizeof
 *	unsafe.Offsetof
 * rewrite with a constant
 */
Node*
unsafenmagic(Node *l, Node *r)
{
	Node *n;
	Sym *s;
	Type *t, *tr;
	long v;
	Val val;

	if(l == N || r == N)
		goto no;
	if(l->op != ONAME)
		goto no;
	s = l->sym;
	if(s == S)
		goto no;
	if(strcmp(s->package, "unsafe") != 0)
		goto no;

	if(strcmp(s->name, "Sizeof") == 0) {
		walktype(r, Erv);
		tr = r->type;
		if(r->op == OLITERAL && r->val.ctype == CTSTR)
			tr = types[TSTRING];
		if(tr == T)
			goto no;
		v = tr->width;
		goto yes;
	}
	if(strcmp(s->name, "Offsetof") == 0) {
		if(r->op != ODOT && r->op != ODOTPTR)
			goto no;
		walktype(r, Erv);
		v = r->xoffset;
		goto yes;
	}
	if(strcmp(s->name, "Alignof") == 0) {
		walktype(r, Erv);
		tr = r->type;
		if(r->op == OLITERAL && r->val.ctype == CTSTR)
			tr = types[TSTRING];
		if(tr == T)
			goto no;

		// make struct { byte; T; }
		t = typ(TSTRUCT);
		t->type = typ(TFIELD);
		t->type->type = types[TUINT8];
		t->type->down = typ(TFIELD);
		t->type->down->type = tr;
		// compute struct widths
		dowidth(t);

		// the offset of T is its required alignment
		v = t->type->down->width;
		goto yes;
	}

no:
	return N;

yes:
	addtop = N;	// any side effects disappear
	val.ctype = CTINT;
	val.u.xval = mal(sizeof(*n->val.u.xval));
	mpmovecfix(val.u.xval, v);
	n = nod(OLITERAL, N, N);
	n->val = val;
	n->type = types[TINT];
	return n;
}
