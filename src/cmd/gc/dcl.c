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
dodcltype(Type *n, Type *t)
{
	Type *nt;

	if(n == T)
		return;
	if(t->sym != S) {
		// botch -- should be a complete deep copy
		nt = typ(Txxx);
		*nt = *t;
		t = nt;
		t->sym = S;
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
		return;
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

	if(dflag())
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
Type*
functype(Node *this, Node *in, Node *out)
{
	Type *t;

	t = typ(TFUNC);

	t->type = dostruct(this, TSTRUCT);
	t->type->down = dostruct(out, TSTRUCT);
	t->type->down->down = dostruct(in, TSTRUCT);

	t->thistuple = listcount(this);
	t->outtuple = listcount(out);
	t->intuple = listcount(in);

	dowidth(t);
	return t;
}

void
funcnam(Type *t, char *nam)
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

Node*
methodname(Node *n, Type *t)
{
	if(isptr[t->etype])
		t = t->type;
	if(t->etype != TSTRUCT)
		goto bad;
	if(t->sym == S)
		goto bad;

	snprint(namebuf, sizeof(namebuf), "%s_%s", t->sym->name, n->sym->name);
	return newname(lookup(namebuf));

bad:
	yyerror("illegal <this> pointer");
	return n;
}

/*
 * add a method, declared as a function,
 * into the structure
 */
void
addmethod(Node *n, Type *pa, Type *t)
{
	Type *f, *d, *p;
	Sym *s;

	if(n->op != ONAME)
		goto bad;
	s = n->sym;
	if(s == S)
		goto bad;
	if(pa == T)
		goto bad;
	if(!isptr[pa->etype])
		goto bad;
	p = pa->type;
	if(p == T)
		goto bad;
	if(p->etype != TSTRUCT)
		goto bad;
	if(p->sym == S)
		goto bad;

	if(p->type == T) {
		n = nod(ODCLFIELD, newname(s), N);
		n->type = t;

		stotype(n, &p->type);
		return;
	}

	d = T;	// last found
	for(f=p->type; f!=T; f=f->down) {
		if(f->etype != TFIELD)
			fatal("addmethod: not TFIELD: %N", f);

		if(strcmp(s->name, f->sym->name) != 0) {
			d = f;
			continue;
		}

		// if a field matches a non-this function
		// then delete it and let it be redeclared
		if(methcmp(t, f->type)) {
			if(d == T) {
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

	if(d == T)
		stotype(n, &p->type);
	else
		stotype(n, &d->down);
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

	// check for same types
	if(on != N) {
		if(eqtype(n->type, on->type, 0)) {
			if(!eqargs(n->type, on->type))
				yyerror("forward declarations not the same: %S", s);
		} else {
			yyerror("redeclare of function: %S", s);
			on = N;
		}
	}

	// check for forward declaration
	if(on == N) {
		// initial declaration or redeclaration
		// declare fun name, argument types and argument names
		funcnam(n->type, s->name);
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

	if(n->type->thistuple > 0) {
		Type *t;
		t = *getthis(n->type);
		addmethod(n->nname, t->type->type, n->type);
	}
}

void
funcargs(Type *t)
{
	Type *n1;
	Iter save;
	int all;

	// declare the this/in arguments
	n1 = funcfirst(&save, t);
	while(n1 != T) {
		if(n1->nname != N)
			addvar(n1->nname, n1->type, PPARAM);
		n1 = funcnext(&save);
	}

	// declare the outgoing arguments
	all = 0;
	n1 = structfirst(&save, getoutarg(t));
	while(n1 != T) {
		if(n1->nname != N && n1->nname->sym->name[0] != '_') {
			addvar(n1->nname, n1->type, PPARAM);
			all |= 1;
		} else
			all |= 2;
		n1 = structnext(&save);
	}
	if(all == 3)
		yyerror("output parameters are all named or not named");

	t->outnamed = 0;
	if(all == 1)
		t->outnamed = 1;
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

	n = listfirst(&save, &n);

loop:
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

	if(n->type->etype == TDARRAY)
		yyerror("type of a structure field cannot be an open array");

	f = typ(TFIELD);
	f->type = n->type;

	if(n->left != N && n->left->op == ONAME) {
		f->nname = n->left;
	} else {
		vargen++;
		snprint(namebuf, sizeof(namebuf), "_e%.3ld", vargen);
		f->nname = newname(lookup(namebuf));
	}
	f->sym = f->nname->sym;

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

	/*
	 * convert a parsed id/type list into
	 * a type for struct/interface/arglist
	 */

	t = typ(et);
	stotype(n, &t->type);
	dowidth(t);
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
	a->forwtype = b->forwtype;
	a->lexical = b->lexical;
	a->undef = b->undef;
	a->vargen = b->vargen;
	a->vblock = b->vblock;
	a->tblock = b->tblock;
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
	block = d->vblock;
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
}

void
markdcl(void)
{
	Sym *d;

	d = push();
	d->name = nil;		// used as a mark in fifo
	d->vblock = block;

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
addvar(Node *n, Type *t, int ctxt)
{
	Dcl *r, *d;
	Sym *s;
	Type *ot;
	Node *on;
	int gen;

	if(n==N || n->sym == S || n->op != ONAME || t == T)
		fatal("addvar: n=%N t=%T nil", n, t);

	ot = t;
	if(isptr[ot->etype])
		ot = ot->type;

	if(ot->etype == TSTRUCT && ot->vargen == 0) {
		vargen++;
		snprint(namebuf, sizeof(namebuf), "_s%.3ld", vargen);
		s = lookup(namebuf);
		addtyp(newtype(s), ot, PEXTERN);
	}

	s = n->sym;
	vargen++;
	gen = vargen;

	r = autodcl;
	if(ctxt == PEXTERN) {
		r = externdcl;
		gen = 0;
	}

	if(s->vblock == block) {
		if(s->oname != N) {
			yyerror("var %S redeclared in this block"
				"\n     previous declaration at %L",
				s, s->oname->lineno);
		} else
			yyerror("var %S redeclared in this block", s);
	}
		
	if(ctxt != PEXTERN)
		pushdcl(s);

	s->vargen = gen;
	s->oname = n;
	s->offset = 0;
	s->vblock = block;

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
addtyp(Type *n, Type *t, int ctxt)
{
	Dcl *r, *d;
	Sym *s;
	Type *f, *ot;

	if(n==T || n->sym == S || t == T)
		fatal("addtyp: n=%T t=%T nil", n, t);

	s = n->sym;

	r = autodcl;
	if(ctxt == PEXTERN) {
		ot = s->otype;
		if(ot != T) {
			// allow nil interface to be
			// redeclared as an interface
			if(ot->etype == TINTER && ot->type == T && t->etype == TINTER) {
				if(dflag())
					print("forew  typ-dcl %S G%ld %T\n", s, s->vargen, t);
				s->otype = t;
				return;
			}
		}
		r = externdcl;
	}

	if(s->tblock == block)
		yyerror("type %S redeclared in this block %d", s, block);

	if(ctxt != PEXTERN)
		pushdcl(s);

	if(t->sym != S)
		warn("addtyp: renaming %S/%lT to %S/%lT", t->sym, t->sym->otype, s, n);

	vargen++;
	s->vargen = vargen;
	s->otype = t;
	s->lexical = LATYPE;
	s->tblock = block;

	t->sym = s;
	t->vargen = vargen;

	for(f=s->forwtype; f!=T; f=f->nforw) {
		if(!isptr[f->etype])
			fatal("addtyp: forward");
		f->type = t;
	}
	s->forwtype = T;

	d = dcl();
	d->dsym = s;
	d->dtype = t;
	d->op = OTYPE;

	r->back->forw = d;
	r->back = d;

	if(dflag()) {
		if(ctxt == PEXTERN)
			print("extern typ-dcl %S G%ld %T\n", s, s->vargen, t);
		else
			print("auto   typ-dcl %S G%ld %T\n", s, s->vargen, t);
	}
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
		n = nod(ONONAME, N, N);
		n->sym = s;
		n->type = T;
		n->addable = 1;
		n->ullman = 0;
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

Type*
forwdcl(Sym *s)
{
	Type *t;

	// this type has no meaning and
	// will cause an error if referenced.
	// it will be patched when/if the
	// type is ever assigned.

	t = typ(TFORW);
	t = ptrto(t);

	t->nforw = s->forwtype;
	s->forwtype = t;
	return t;
}
