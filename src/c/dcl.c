// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include	"go.h"
#include	"y.tab.h"

void
dodclvar(Node *n, Node *t)
{

loop:
	if(n == N)
		return;

	if(n->op == OLIST) {
		dodclvar(n->left, t);
		n = n->right;
		goto loop;
	}

	addvar(n, t, dclcontext);
}

void
dodcltype(Node *n, Node *t)
{

loop:
	if(n == N)
		return;

	if(n->op == OLIST) {
		dodcltype(n->left, t);
		n = n->right;
		goto loop;
	}

	addtyp(n, t, dclcontext);
}

void
dodclconst(Node *n, Node *e)
{
	Sym *s;
	Dcl *r, *d;

loop:
	if(n == N)
		return;
	if(n->op == OLIST) {
		dodclconst(n->left, e);
		n = n->right;
		goto loop;
	}

	if(n->op != ONAME)
		fatal("dodclconst: not a name");

	if(e->op != OLITERAL) {
		yyerror("expression must be a constant");
		goto loop;
	}
	s = n->sym;

	s->oconst = e;
	s->lexical = LACONST;

	r = autodcl;
	if(dclcontext == PEXTERN)
		r = externdcl;

	d = dcl();
	d->dsym = s;
	d->dnode = e;
	d->op = OCONST;

	r->back->forw = d;
	r->back = d;

	if(debug['d'])
		print("const-dcl %S %N\n", n->sym, n->sym->oconst);
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
Node*
functype(Node *this, Node *in, Node *out)
{
	Node *t;

	t = nod(OTYPE, N, N);
	t->etype = TFUNC;

	t->type = dostruct(this, TSTRUCT);
	t->type->down = dostruct(out, TSTRUCT);
	t->type->down->down = dostruct(in, TSTRUCT);

	t->thistuple = listcount(this);
	t->outtuple = listcount(out);
	t->intuple = listcount(in);

	return t;
}

void
funcnam(Node *t, char *nam)
{
	Node *n;
	Sym *s;
	char buf[100];

	if(nam == nil) {
		vargen++;
		snprint(buf, sizeof(buf), "_f%.3ld", vargen);
		nam = buf;
	}

	if(t->etype != TFUNC)
		fatal("funcnam: not func %T\n", t);

	if(t->thistuple > 0) {
		vargen++;
		snprint(namebuf, sizeof(namebuf), "_t%.3ld", vargen);
		s = lookup(namebuf);
		addtyp(newtype(s), t->type, PEXTERN);
		n = newname(s);
		n->vargen = vargen;
		t->type->nname = n;
	}
	if(t->outtuple > 0) {
		vargen++;
		snprint(namebuf, sizeof(namebuf), "_o%.3ld", vargen);
		s = lookup(namebuf);
		addtyp(newtype(s), t->type->down, PEXTERN);
		n = newname(s);
		n->vargen = vargen;
		t->type->down->nname = n;
	}
	if(t->intuple > 0) {
		vargen++;
		snprint(namebuf, sizeof(namebuf), "_i%.3ld", vargen);
		s = lookup(namebuf);
		addtyp(newtype(s), t->type->down->down, PEXTERN);
		n = newname(s);
		n->vargen = vargen;
		t->type->down->down->nname = n;
	}
}

int
methcmp(Node *t1, Node *t2)
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
		if(t1 == N || t2 == N)
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

/*
 * add a method, declared as a function,
 * into the structure
 */
void
addmethod(Node *n, Node *pa, Node *t)
{
	Node *p, *f, *d;
	Sym *s;

	if(n->op != ONAME)
		goto bad;
	s = n->sym;
	if(s == S)
		goto bad;
	if(pa == N)
		goto bad;
	if(pa->etype != TPTR)
		goto bad;
	p = pa->type;
	if(p == N)
		goto bad;
	if(p->etype != TSTRUCT)
		goto bad;
	if(p->sym == S)
		goto bad;

	if(p->type == N) {
		n = nod(ODCLFIELD, newname(s), N);
		n->type = t;

		stotype(n, &p->type, p);
		return;
	}

	d = N;	// last found
	for(f=p->type; f!=N; f=f->down) {
		if(f->etype != TFIELD)
			fatal("addmethod: not TFIELD: %N", f);

		if(strcmp(s->name, f->sym->name) != 0) {
			d = f;
			continue;
		}

		// if a field matches a non-this function
		// then delete it and let it be redeclared
		if(methcmp(t, f->type)) {
			if(d == N) {
				p->type = f->down;
				continue;
			}
			d->down = f->down;
			continue;
		}
		if(!eqtype(t, f->type, 0))
			yyerror("field redeclared as method: %S", s);
		return;
	}

	n = nod(ODCLFIELD, newname(s), N);
	n->type = t;

	if(d == N)
		stotype(n, &p->type, p);
	else
		stotype(n, &d->down, p);
	return;

bad:
	yyerror("unknown method pointer: %T", pa);
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

	// check for foreward declaration
	if(on == N || !eqtype(n->type, on->type, 0)) {
		// initial declaration or redeclaration
		// declare fun name, argument types and argument names
		funcnam(n->type, s->name);
		n->nname->type = n->type;
		if(n->type->thistuple == 0)
			addvar(n->nname, n->type, PEXTERN);
	} else {
		// identical redeclaration
		// steal previous names
		n->nname = on;
		n->type = on->type;
		n->sym = s;
		s->oname = n;
		if(debug['d'])
			print("forew  var-dcl %S %T\n", n->sym, n->type);
	}

	// change the declaration context from extern to auto
	autodcl = dcl();
	autodcl->back = autodcl;

	if(dclcontext != PEXTERN)
		fatal("funchdr: dclcontext");
	dclcontext = PAUTO;
	markdcl("func");

	funcargs(n->type);
	if(n->type->thistuple > 0) {
		Node *n1;
		n1 = *getthis(n->type);
		addmethod(n->nname, n1->type->type, n->type);
	}
}

void
funcargs(Node *t)
{
	Node *n1;
	Iter save;

	// declare the this argument
	n1 = structfirst(&save, getthis(t));
	if(n1 != N) {
		if(n1->nname != N)
			addvar(n1->nname, n1->type, PAUTO);
	}

	// declare the incoming arguments
	n1 = structfirst(&save, getinarg(t));
	while(n1 != N) {
		if(n1->nname != N)
			addvar(n1->nname, n1->type, PAUTO);
		n1 = structnext(&save);
	}

	// declare the outgoing arguments
//	n1 = structfirst(&save, getoutarg(t));
//	while(n1 != N) {
//		n1->left = newname(n1->sym);
//		if(n1->nname != N)
//			addvar(n1->nname, n1->type, PAUTO);
//		n1 = structnext(&save);
//	}
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
	dclcontext = PEXTERN;
	popdcl("func");
}

/*
 * turn a parsed struct into a type
 */
Node**
stotype(Node *n, Node **t, Node *uber)
{
	Node *f;
	Iter save;

	n = listfirst(&save, &n);

loop:
	if(n == N) {
		*t = N;
		return t;
	}

	if(n->op == OLIST) {
		// recursive because it can be lists of lists
		t = stotype(n, t, uber);
		goto next;
	}

	if(n->op != ODCLFIELD || n->type == N)
		fatal("stotype: oops %N\n", n);

	if(n->type->etype == TDARRAY)
		yyerror("type of a structure field cannot be an open array");

	f = nod(OTYPE, N, N);
	f->etype = TFIELD;
	f->type = n->type;
	f->uberstruct = uber;

	if(n->left != N && n->left->op == ONAME) {
		f->nname = n->left;
	} else {
		vargen++;
		snprint(namebuf, sizeof(namebuf), "_e%.3ld", vargen);
		f->nname = newname(lookup(namebuf));
	}
	f->sym = f->nname->sym;
	f->nname->uberstruct = uber;	// can reach parent from element

	*t = f;
	t = &f->down;

next:
	n = listnext(&save);
	goto loop;
}

Node*
dostruct(Node *n, int et)
{
	Node *t;

	/*
	 * convert a parsed id/type list into
	 * a type for struct/interface/arglist
	 */

	t = nod(OTYPE, N, N);
	stotype(n, &t->type, t);
	t->etype = et;
	return t;
}

Node*
sortinter(Node *n)
{
	return n;
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
	a->forwtype = b->forwtype;
	a->lexical = b->lexical;
	a->undef = b->undef;
	a->vargen = b->vargen;
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
popdcl(char *why)
{
	Sym *d, *s;

//	if(debug['d'])
//		print("revert\n");
	for(d=dclstack; d!=S; d=d->link) {
		if(d->name == nil)
			break;
		s = pkglookup(d->name, d->package);
		dcopy(s, d);
		if(debug['d'])
			print("\t%ld pop %S\n", curio.lineno, s);
	}
	if(d == S)
		fatal("popdcl: no mark");
	if(strcmp(why, d->package) != 0)
		fatal("popdcl: pushed as %s poped as %s", d->package, why);
	dclstack = d->link;
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
		if(debug['d'])
			print("\t%ld pop %S\n", curio.lineno, s);
	}
	if(d == S)
		fatal("poptodcl: no mark");
}

void
markdcl(char *why)
{
	Sym *d;

	d = push();
	d->name = nil;		// used as a mark in fifo
	d->package = why;	// diagnostic for unmatched
//	if(debug['d'])
//		print("markdcl\n");
}

void
markdclstack(void)
{
	Sym *d, *s;

	markdcl("fnlit");

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
addvar(Node *n, Node *t, int ctxt)
{
	Dcl *r, *d;
	Sym *s;
	Node *on;
	int gen;

	if(n==N || n->sym == S || n->op != ONAME || t == N)
		fatal("addvar: n=%N t=%N nil", n, t);

	on = t;
	if(on->etype == TPTR)
		on = on->type;
	if(on->etype == TSTRUCT && on->vargen == 0) {
		vargen++;
		snprint(namebuf, sizeof(namebuf), "_s%.3ld", vargen);
		addtyp(newtype(lookup(namebuf)), on, PEXTERN);
	}

	s = n->sym;
	vargen++;
	gen = vargen;

	r = autodcl;
	if(ctxt == PEXTERN) {
		on = s->oname;
		if(on != N) {
			if(eqtype(t, on->type, 0)) {
				warn("%S redeclared", s);
				return;
			}
			yyerror("%S redeclared (%T %T)", s,
				on->type, t);
		}
		r = externdcl;
		gen = 0;
	}

	pushdcl(s);
	s->vargen = gen;
	s->oname = n;

	n->type = t;
	n->vargen = gen;

	d = dcl();
	d->dsym = s;
	d->dnode = n;
	d->op = ONAME;

	r->back->forw = d;
	r->back = d;

	if(debug['d']) {
		if(ctxt == PEXTERN)
			print("extern var-dcl %S G%ld %T\n", s, s->vargen, t);
		else
			print("auto   var-dcl %S G%ld %T\n", s, s->vargen, t);
	}
}

void
addtyp(Node *n, Node *t, int ctxt)
{
	Dcl *r, *d;
	Sym *s;
	Node *f, *ot;

	if(n==N || n->sym == S || n->op != OTYPE || t == N)
		fatal("addtyp: n=%N t=%N nil", n, t);

	s = n->sym;

	r = autodcl;
	if(ctxt == PEXTERN) {
		ot = s->otype;
		if(ot != N) {
			// allow nil interface to be
			// redeclared as an interface
			if(ot->etype == TINTER && ot->type == N && t->etype == TINTER) {
				if(debug['d'])
					print("forew  typ-dcl %S G%ld %T\n", s, s->vargen, t);
				s->otype = t;
				return;
			}
			if(eqtype(t, ot, 0)) {
				warn("%S redeclared", s);
				return;
			}
			yyerror("%S redeclared (%T %T)", s,
				ot, t);
		}
		r = externdcl;
	}

	pushdcl(s);
	vargen++;
	s->vargen = vargen;
	s->otype = t;
	s->lexical = LATYPE;

	if(t->sym != S)
		warn("addtyp: renaming %S to %S", t->sym, s);

	t->sym = s;
	t->vargen = vargen;

	for(f=s->forwtype; f!=N; f=f->nforw) {
		if(f->op != OTYPE && f->etype != TPTR)
			fatal("addtyp: foreward");
		f->type = t;
	}
	s->forwtype = N;

	d = dcl();
	d->dsym = s;
	d->dnode = t;
	d->op = OTYPE;

	r->back->forw = d;
	r->back = d;

	if(debug['d']) {
		if(ctxt == PEXTERN)
			print("extern typ-dcl %S G%ld %T\n", s, s->vargen, t);
		else
			print("auto   typ-dcl %S G%ld %T\n", s, s->vargen, t);
	}
}

/*
 * make a new variable
 */
Node*
tempname(Node *t)
{
	Sym *s;
	Node *n;

	if(t == N) {
		yyerror("tempname called with nil type");
		t = types[TINT32];
	}

	s = lookup("!tmpname!");
	n = newname(s);
	dodclvar(n, t);
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
	n->type = N;
	n->addable = 1;
	n->ullman = 0;
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
		yyerror("%S undefined", s);
		n = newname(s);
		dodclvar(n, types[TINT32]);
	}
	return n;
}

/*
 * same for types
 */
Node*
newtype(Sym *s)
{
	Node *n;

	n = nod(OTYPE, N, N);
	n->etype = TFORW;
	n->sym = s;
	n->type = N;
	return n;
}

Node*
oldtype(Sym *s)
{
	Node *n;

	n = s->otype;
	if(n == N)
		fatal("%S not a type", s); // cant happen
	return n;
}

Node*
forwdcl(Sym *s)
{
	Node *n;

	// this type has no meaning and
	// will cause an error if referenced.
	// it will be patched when/if the
	// type is ever assigned.
	n = nod(OTYPE, N, N);
	n->etype = TFORW;
	n = ptrto(n);

	n->nforw = s->forwtype;
	s->forwtype = n;
	return n;
}
