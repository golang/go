// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include	"go.h"
#include	"y.tab.h"

void
markexport(Node *n)
{
	Sym *s;
	Dcl *d, *r;

loop:
	if(n == N)
		return;

	if(n->op == OLIST) {
		markexport(n->left);
		n = n->right;
		goto loop;
	}

	if(n->op != OEXPORT)
		fatal("markexport: op no OEXPORT: %O", n->op);

	s = n->sym;
	if(n->psym != S)
		s = pkglookup(n->sym->name, n->psym->name);

	if(s->export != 0)
		return;
	s->export = 1;

	d = mal(sizeof(*d));
	d->dsym = s;
	d->dnode = N;
	d->lineno = lineno;

	r = exportlist;
	d->back = r->back;
	r->back->forw = d;
	r->back = d;
}

void
reexport(Type *t)
{
	Sym *s;

	if(t == T)
		fatal("reexport: type nil\n");

	s = t->sym;
	if(s == S/* || s->name[0] == '_'*/) {
		exportgen++;
		snprint(namebuf, sizeof(namebuf), "_e%.3ld", exportgen);
		s = lookup(namebuf);
		s->lexical = LATYPE;
		s->otype = t;
		t->sym = s;
	}
	dumpexporttype(s);
}

void
dumpexportconst(Sym *s)
{
	Node *n;
	Type *t;

	if(s->exported != 0)
		return;
	s->exported = 1;

	n = s->oconst;
	if(n == N || n->op != OLITERAL)
		fatal("dumpexportconst: oconst nil: %S\n", s);

	t = n->type;	// may or may not be specified
	if(t != T)
		reexport(t);

	Bprint(bout, "\tconst ");
	if(s->export != 0)
		Bprint(bout, "!");
	Bprint(bout, "%lS ", s);
	if(t != T)
		Bprint(bout, "%lS ", t->sym);

	switch(n->val.ctype) {
	default:
		fatal("dumpexportconst: unknown ctype: %S\n", s);
	case CTINT:
	case CTSINT:
	case CTUINT:
	case CTBOOL:
		Bprint(bout, "0x%llux\n", n->val.vval);
		break;
	case CTFLT:
		Bprint(bout, "%.17e\n", n->val.dval);
		break;
	case CTSTR:
		Bprint(bout, "\"%Z\"\n", n->val.sval);
		break;
	}
}

void
dumpexportvar(Sym *s)
{
	Node *n;
	Type *t;

	if(s->exported != 0)
		return;
	s->exported = 1;

	n = s->oname;
	if(n == N || n->type == T)
		fatal("dumpexportvar: oname nil: %S\n", s);

	t = n->type;
	reexport(t);

	Bprint(bout, "\tvar ");
	if(s->export != 0)
		Bprint(bout, "!");
	Bprint(bout, "%lS %lS\n", s, t->sym);
}

void
dumpexporttype(Sym *s)
{
	Type *t, *f;
	Sym *ts;
	int et;

	if(s->exported != 0)
		return;
	s->exported = 1;

	t = s->otype;
	if(t == T)
		fatal("dumpexporttype: otype nil: %S\n", s);
	if(t->sym != s)
		fatal("dumpexporttype: cross reference: %S\n", s);

	et = t->etype;
	switch(et) {
	default:
		if(et < 0 || et >= nelem(types) || types[et] == T)
			fatal("dumpexporttype: basic type: %S %E\n", s, et);
		/* type 5 */
		Bprint(bout, "\ttype %lS %d\n", s, et);
		break;

	case TARRAY:
		reexport(t->type);

		/* type 2 */
		Bprint(bout, "\ttype ");
		if(s->export != 0)
			Bprint(bout, "!");
		Bprint(bout, "%lS [%lud] %lS\n", s, t->bound, t->type->sym);
		break;

	case TPTR32:
	case TPTR64:
		reexport(t->type);

		/* type 6 */
		Bprint(bout, "\ttype ");
		if(s->export != 0)
			Bprint(bout, "!");
		Bprint(bout, "%lS *%lS\n", s, t->type->sym);
		break;

	case TFUNC:
		for(f=t->type; f!=T; f=f->down) {
			if(f->etype != TSTRUCT)
				fatal("dumpexporttype: funct not field: %T\n", f);
			reexport(f);
		}

		/* type 3 */
		Bprint(bout, "\ttype ");
		if(s->export != 0)
			Bprint(bout, "!");
		Bprint(bout, "%lS (", s);
		for(f=t->type; f!=T; f=f->down) {
			if(f != t->type)
				Bprint(bout, " ");
			Bprint(bout, "%lS", f->sym);
		}
		Bprint(bout, ")\n");
		break;

	case TSTRUCT:
	case TINTER:
		for(f=t->type; f!=T; f=f->down) {
			if(f->etype != TFIELD)
				fatal("dumpexporttype: funct not field: %lT\n", f);
			reexport(f->type);
		}

		/* type 4 */
		Bprint(bout, "\ttype ");
		if(s->export)
			Bprint(bout, "!");
		Bprint(bout, "%lS %c", s, (et==TSTRUCT)? '{': '<');
		for(f=t->type; f!=T; f=f->down) {
			ts = f->type->sym;
			if(f != t->type)
				Bprint(bout, " ");
			Bprint(bout, "%s %lS", f->sym->name, ts);
		}
		Bprint(bout, "%c\n", (et==TSTRUCT)? '}': '>');
		break;

	case TMAP:
		reexport(t->type);
		reexport(t->down);

		/* type 6 */
		Bprint(bout, "\ttype ");
		if(s->export != 0)
			Bprint(bout, "!");
		Bprint(bout, "%lS [%lS] %lS\n", s, t->down->sym, t->type->sym);
		break;
	}
}

void
dumpe(Sym *s)
{
	switch(s->lexical) {
	default:
		yyerror("unknown export symbol: %S\n", s, s->lexical);
		break;
	case LPACK:
		yyerror("package export symbol: %S\n", s);
		break;
	case LATYPE:
	case LBASETYPE:
		dumpexporttype(s);
		break;
	case LNAME:
		dumpexportvar(s);
		break;
	case LACONST:
		dumpexportconst(s);
		break;
	}
}

void
dumpexport(void)
{
	Dcl *d;
	long lno;

	lno = dynlineno;

	Bprint(bout, "   import\n");
	Bprint(bout, "   ((\n");

	Bprint(bout, "    package %s\n", package);

	// print it depth first
	for(d=exportlist->forw; d!=D; d=d->forw) {
		dynlineno = d->lineno;
		dumpe(d->dsym);
	}

	Bprint(bout, "   ))\n");

	dynlineno = lno;
}

/*
 * ******* import *******
 */
void
renamepkg(Node *n)
{
	if(n->psym == pkgimportname)
		if(pkgmyname != S)
			n->psym = pkgmyname;

	if(n->psym->lexical != LPACK) {
		warn("%S is becoming a package behind your back", n->psym);
		n->psym->lexical = LPACK;
	}
}

Sym*
getimportsym(Node *ss)
{
	char *pkg;
	Sym *s;

	if(ss->op != OIMPORT)
		fatal("getimportsym: oops1 %N\n", ss);

	pkg = ss->psym->name;
	s = pkglookup(ss->sym->name, pkg);

	/* botch - need some diagnostic checking for the following assignment */
	s->opackage = ss->osym->name;
	return s;
}

Type*
importlooktype(Node *n)
{
	Sym *s;

	s = getimportsym(n);
	if(s->otype == T)
		fatal("importlooktype: oops2 %S\n", s);
	return s->otype;
}

Type**
importstotype(Node *fl, Type **t, Type *uber)
{
	Type *f;
	Iter save;
	Node *n;

	n = listfirst(&save, &fl);

loop:
	if(n == N) {
		*t = T;
		return t;
	}
	f = typ(TFIELD);
	f->type = importlooktype(n);

	if(n->fsym != S) {
		f->nname = newname(n->fsym);
	} else {
		vargen++;
		snprint(namebuf, sizeof(namebuf), "_m%.3ld", vargen);
		f->nname = newname(lookup(namebuf));
	}
	f->sym = f->nname->sym;

	*t = f;
	t = &f->down;

	n = listnext(&save);
	goto loop;
}

int
importcount(Type *t)
{
	int i;
	Type *f;

	if(t == T || t->etype != TSTRUCT)
		fatal("importcount: not a struct: %N", t);

	i = 0;
	for(f=t->type; f!=T; f=f->down)
		i = i+1;
	return i;
}

void
importfuncnam(Type *t)
{
	Node *n;
	Type *t1;

	if(t->etype != TFUNC)
		fatal("importfuncnam: not func %T\n", t);

	if(t->thistuple > 0) {
		t1 = t->type;
		if(t1->sym == S)
			fatal("importfuncnam: no this");
		n = newname(t1->sym);
		vargen++;
		n->vargen = vargen;
		t1->nname = n;
	}
	if(t->outtuple > 0) {
		t1 = t->type->down;
		if(t1->sym == S)
			fatal("importfuncnam: no output");
		n = newname(t1->sym);
		vargen++;
		n->vargen = vargen;
		t1->nname = n;
	}
	if(t->intuple > 0) {
		t1 = t->type->down->down;
		if(t1->sym == S)
			fatal("importfuncnam: no input");
		n = newname(t1->sym);
		vargen++;
		n->vargen = vargen;
		t1->nname = n;
	}
}

void
importaddtyp(Node *ss, Type *t)
{
	Sym *s;

	s = getimportsym(ss);
	if(s->otype == T) {
		addtyp(newtype(s), t, PEXTERN);
		return;
	}
	if(!eqtype(t, s->otype, 0)) {
		print("redeclaring %S %lT => %lT\n", s, s->otype, t);
		addtyp(newtype(s), t, PEXTERN);
		return;
	}
	print("sametype %S %lT => %lT\n", s, s->otype, t);
}

/*
 * LCONST importsym LITERAL
 * untyped constant
 */
void
doimportc1(Node *ss, Val *v)
{
	Node *n;
	Sym *s;

	n = nod(OLITERAL, N, N);
	n->val = *v;

	s = getimportsym(ss);
	if(s->oconst == N) {
		// botch sould ask if already declared the same
		dodclconst(newname(s), n);
	}
}

/*
 * LCONST importsym importsym LITERAL
 * typed constant
 */
void
doimportc2(Node *ss, Node *st, Val *v)
{
	Node *n;
	Type *t;
	Sym *s;

	n = nod(OLITERAL, N, N);
	n->val = *v;

	t = importlooktype(st);
	n->type = t;

	s = getimportsym(ss);
	if(s->oconst == N) {
		// botch sould ask if already declared the same
		dodclconst(newname(s), n);
	}
}

/*
 * LVAR importsym importsym
 * variable
 */
void
doimportv1(Node *ss, Node *st)
{
	Type *t;
	Sym *s;

	t = importlooktype(st);
	s = getimportsym(ss);
	if(s->oname == N || !eqtype(t, s->oname->type, 0)) {
		addvar(newname(s), t, dclcontext);
	}
}

/*
 * LTYPE importsym [ importsym ] importsym
 * array type
 */
void
doimport1(Node *ss, Node *si, Node *st)
{
	Type *t;
	Sym *s;

	t = typ(TMAP);
	s = pkglookup(si->sym->name, si->psym->name);
	t->down = s->otype;
	s = pkglookup(st->sym->name, st->psym->name);
	t->type = s->otype;

	importaddtyp(ss, t);
}

/*
 * LTYPE importsym [ LLITERAL ] importsym
 * array type
 */
void
doimport2(Node *ss, Val *b, Node *st)
{
	Type *t;
	Sym *s;

	t = typ(TARRAY);
	t->bound = b->vval;
	s = pkglookup(st->sym->name, st->psym->name);
	t->type = s->otype;

	importaddtyp(ss, t);
}

/*
 * LTYPE importsym '(' importsym_list ')'
 * function/method type
 */
void
doimport3(Node *ss, Node *n)
{
	Type *t;

	t = typ(TFUNC);

	t->type = importlooktype(n->left);
	t->type->down = importlooktype(n->right->left);
	t->type->down->down = importlooktype(n->right->right);

	t->thistuple = importcount(t->type);
	t->outtuple = importcount(t->type->down);
	t->intuple = importcount(t->type->down->down);

	importfuncnam(t);

	importaddtyp(ss, t);
}

/*
 * LTYPE importsym '{' importsym_list '}'
 * structure type
 */
void
doimport4(Node *ss, Node *n)
{
	Type *t;

	t = typ(TSTRUCT);
	importstotype(n, &t->type, t);

	importaddtyp(ss, t);
}

/*
 * LTYPE importsym LLITERAL
 * basic type
 */
void
doimport5(Node *ss, Val *v)
{
	int et;
	Type *t;

	et = v->vval;
	if(et <= 0 || et >= nelem(types) || types[et] == T)
		fatal("doimport5: bad type index: %E\n", et);

	t = typ(et);
	t->sym = S;

	importaddtyp(ss, t);
}

/*
 * LTYPE importsym * importsym
 * pointer type
 */
void
doimport6(Node *ss, Node *st)
{
	Type *t;
	Sym *s;

	s = pkglookup(st->sym->name, st->psym->name);
	t = s->otype;
	if(t == T)
		t = forwdcl(s);
	else
		t = ptrto(t);

	importaddtyp(ss, t);
}

/*
 * LTYPE importsym '<' importsym '>'
 * interface type
 */
void
doimport7(Node *ss, Node *n)
{
	Type *t;

	t = typ(TINTER);
	importstotype(n, &t->type, t);

	importaddtyp(ss, t);
}
