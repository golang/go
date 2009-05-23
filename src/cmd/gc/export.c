// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include	"go.h"
#include	"y.tab.h"

void dumpsym(Sym*);

void
addexportsym(Sym *s)
{
	Dcl *d, *r;

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
exportsym(Sym *s)
{
	if(s == S)
		return;
	if(s->export != 0) {
		if(s->export != 1)
			yyerror("export/package mismatch: %S", s);
		return;
	}
	s->export = 1;

	addexportsym(s);
}

void
packagesym(Sym *s)
{
	if(s == S)
		return;
	if(s->export != 0) {
		if(s->export != 2)
			yyerror("export/package mismatch: %S", s);
		return;
	}
	s->export = 2;

	addexportsym(s);
}

int
exportname(char *s)
{
	Rune r;

	if((uchar)s[0] < Runeself)
		return 'A' <= s[0] && s[0] <= 'Z';
	chartorune(&r, s);
	return isupperrune(r);
}

void
autoexport(Sym *s)
{
	if(s == S)
		return;
	if(dclcontext != PEXTERN)
		return;
	if(exportname(s->name)) {
		exportsym(s);
	} else {
		packagesym(s);
	}
}

void
dumpprereq(Type *t)
{
	if(t == T)
		return;

	if(t->printed || t == types[t->etype])
		return;
	t->printed = 1;

	if(t->sym != S && t->etype != TFIELD)
		dumpsym(t->sym);
	dumpprereq(t->type);
	dumpprereq(t->down);
}

void
dumpexportconst(Sym *s)
{
	Node *n;
	Type *t;

	n = s->oconst;
	if(n == N || n->op != OLITERAL)
		fatal("dumpexportconst: oconst nil: %S", s);

	t = n->type;	// may or may not be specified
	if(t != T)
		dumpprereq(t);

	Bprint(bout, "\t");
	Bprint(bout, "const %lS", s);
	if(t != T && t->etype != TIDEAL)
		Bprint(bout, " %#T", t);
	Bprint(bout, " = ");

	switch(n->val.ctype) {
	default:
		fatal("dumpexportconst: unknown ctype: %S", s);
	case CTINT:
		Bprint(bout, "%B\n", n->val.u.xval);
		break;
	case CTBOOL:
		if(n->val.u.bval)
			Bprint(bout, "true\n");
		else
			Bprint(bout, "false\n");
		break;
	case CTFLT:
		Bprint(bout, "%F\n", n->val.u.fval);
		break;
	case CTSTR:
		Bprint(bout, "\"%Z\"\n", n->val.u.sval);
		break;
	}
}

void
dumpexportvar(Sym *s)
{
	Node *n;
	Type *t;

	n = s->oname;
	if(n == N || n->type == T) {
		yyerror("variable exported but not defined: %S", s);
		return;
	}

	t = n->type;
	dumpprereq(t);

	Bprint(bout, "\t");
	if(t->etype == TFUNC && n->class == PFUNC)
		Bprint(bout, "func %lS %#hhT", s, t);
	else
		Bprint(bout, "var %lS %#T", s, t);
	Bprint(bout, "\n");
}

void
dumpexporttype(Sym *s)
{
	dumpprereq(s->otype);
	Bprint(bout, "\t");
	switch (s->otype->etype) {
	case TFORW:
	case TFORWSTRUCT:
	case TFORWINTER:
		yyerror("export of incomplete type %T", s->otype);
		return;
	}
	Bprint(bout, "type %#T %l#T\n",  s->otype, s->otype);
}

void
dumpsym(Sym *s)
{
	Type *f;

	if(s->exported != 0)
		return;
	s->exported = 1;

	switch(s->lexical) {
	default:
		yyerror("unknown export symbol: %S", s);
		break;
	case LPACK:
		yyerror("package export symbol: %S", s);
		break;
	case LATYPE:
		// TODO(rsc): sort methods by name
		for(f=s->otype->method; f!=T; f=f->down)
			dumpprereq(f);

		dumpexporttype(s);
		for(f=s->otype->method; f!=T; f=f->down)
			Bprint(bout, "\tfunc (%#T) %hS %#hhT\n",
				f->type->type->type, f->sym, f->type);
		break;
	case LNAME:
		if(s->oconst)
			dumpexportconst(s);
		else
			dumpexportvar(s);
		break;
	}
}

void
dumptype(Type *t)
{
	// no need to re-dump type if already exported
	if(t->printed)
		return;

	// no need to dump type if it's not ours (was imported)
	if(t->sym != S && t->sym->otype == t && !t->local)
		return;

	Bprint(bout, "type %#T %l#T\n",  t, t);
}

void
dumpexport(void)
{
	Dcl *d;
	int32 lno;

	lno = lineno;

	Bprint(bout, "   import\n");
	Bprint(bout, "\n$$  // exports\n");

	Bprint(bout, "    package %s\n", package);

	for(d=exportlist->forw; d!=D; d=d->forw) {
		lineno = d->lineno;
		dumpsym(d->dsym);
	}

	Bprint(bout, "\n$$  // local types\n");

	for(d=typelist->forw; d!=D; d=d->forw) {
		lineno = d->lineno;
		dumptype(d->dtype);
	}

	Bprint(bout, "\n$$\n");

	lineno = lno;
}

/*
 * import
 */

/*
 * look up and maybe declare pkg.name, which should match lexical
 */
Sym*
pkgsym(char *name, char *pkg, int lexical)
{
	Sym *s;

	s = pkglookup(name, pkg);
	switch(lexical) {
	case LATYPE:
		if(s->oname)
			yyerror("%s.%s is not a type", name, pkg);
		break;
	case LNAME:
		if(s->otype)
			yyerror("%s.%s is not a name", name, pkg);
		break;
	}
	s->lexical = lexical;
	return s;
}

/*
 * return the sym for ss, which should match lexical
 */
Sym*
importsym(Node *ss, int lexical)
{
	Sym *s;

	if(ss->op != OIMPORT)
		fatal("importsym: oops1 %N", ss);

	s = pkgsym(ss->sym->name, ss->psym->name, lexical);
	/* TODO botch - need some diagnostic checking for the following assignment */
	if(exportname(ss->sym->name))
		s->export = 1;
	else
		s->export = 2;	// package scope
	s->imported = 1;
	return s;
}

/*
 * return the type pkg.name, forward declaring if needed
 */
Type*
pkgtype(char *name, char *pkg)
{
	Sym *s;
	Type *t;

	// botch
	// s = pkgsym(name, pkg, LATYPE);
	Node *n;
	n = nod(OIMPORT, N, N);
	n->sym = lookup(name);
	n->psym = lookup(pkg);
	s = importsym(n, LATYPE);

	if(s->otype == T) {
		t = typ(TFORW);
		t->sym = s;
		s->otype = t;
	}
	return s->otype;
}

static int
mypackage(Node *ss)
{
	// we import all definitions for sys.
	// lowercase ones can only be used by the compiler.
	return strcmp(ss->psym->name, package) == 0
		|| strcmp(ss->psym->name, "sys") == 0;
}

void
importconst(Node *ss, Type *t, Node *n)
{
	Sym *s;

	if(!exportname(ss->sym->name) && !mypackage(ss))
		return;

	convlit(n, t);
	s = importsym(ss, LNAME);
	if(s->oconst != N) {
		// TODO: check if already the same.
		return;
	}

	dodclconst(newname(s), n);

	if(debug['E'])
		print("import const %S\n", s);
}

void
importvar(Node *ss, Type *t, int ctxt)
{
	Sym *s;

	if(!exportname(ss->sym->name) && !mypackage(ss))
		return;

	s = importsym(ss, LNAME);
	if(s->oname != N) {
		if(cvttype(t, s->oname->type))
			return;
		warn("redeclare import var %S from %T to %T",
			s, s->oname->type, t);
	}
	checkwidth(t);
	addvar(newname(s), t, ctxt);

	if(debug['E'])
		print("import var %S %lT\n", s, t);
}

void
importtype(Node *ss, Type *t)
{
	Sym *s;

	s = importsym(ss, LATYPE);
	if(s->otype != T) {
		if(cvttype(t, s->otype))
			return;
		if(s->otype->etype != TFORW) {
			warn("redeclare import type %S from %lT to %lT",
				s, s->otype, t);
			s->otype = typ(0);
		}
	}
	if(s->otype == T)
		s->otype = typ(0);
	*s->otype = *t;
	s->otype->sym = s;
	checkwidth(s->otype);

	if(debug['E'])
		print("import type %S %lT\n", s, t);
}

void
importmethod(Sym *s, Type *t)
{
	checkwidth(t);
	addmethod(newname(s), t, 0);
}

/*
 * ******* import *******
 */

void
checkimports(void)
{
	Sym *s;
	Type *t, *t1;
	uint32 h;
	int et;

return;

	for(h=0; h<NHASH; h++)
	for(s = hash[h]; s != S; s = s->link) {
		t = s->otype;
		if(t == T)
			continue;

		et = t->etype;
		switch(t->etype) {
		case TFORW:
			print("ci-1: %S %lT\n", s, t);
			break;

		case TPTR32:
		case TPTR64:
			if(t->type == T) {
				print("ci-2: %S %lT\n", s, t);
				break;
			}

			t1 = t->type;
			if(t1 == T) {
				print("ci-3: %S %lT\n", s, t1);
				break;
			}

			et = t1->etype;
			if(et == TFORW) {
				print("%L: ci-4: %S %lT\n", lineno, s, t);
				break;
			}
			break;
		}
	}
}

