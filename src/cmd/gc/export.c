// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include	<u.h>
#include	<libc.h>
#include	"go.h"
#include	"y.tab.h"

static	void	dumpsym(Sym*);
static	void	dumpexporttype(Type*);
static	void	dumpexportvar(Sym*);
static	void	dumpexportconst(Sym*);

void
exportsym(Node *n)
{
	if(n == N || n->sym == S)
		return;
	if(n->sym->flags & (SymExport|SymPackage)) {
		if(n->sym->flags & SymPackage)
			yyerror("export/package mismatch: %S", n->sym);
		return;
	}
	n->sym->flags |= SymExport;

	exportlist = list(exportlist, n);
}

static void
packagesym(Node *n)
{
	if(n == N || n->sym == S)
		return;
	if(n->sym->flags & (SymExport|SymPackage)) {
		if(n->sym->flags & SymExport)
			yyerror("export/package mismatch: %S", n->sym);
		return;
	}
	n->sym->flags |= SymPackage;

	exportlist = list(exportlist, n);
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

static int
initname(char *s)
{
	return strcmp(s, "init") == 0;
}

void
autoexport(Node *n, int ctxt)
{
	if(n == N || n->sym == S)
		return;
	if((ctxt != PEXTERN && ctxt != PFUNC) || dclcontext != PEXTERN)
		return;
	if(n->ntype && n->ntype->op == OTFUNC && n->ntype->left)	// method
		return;
	if(exportname(n->sym->name) || initname(n->sym->name))
		exportsym(n);
	else
		packagesym(n);
}

static void
dumppkg(Pkg *p)
{
	char *suffix;

	if(p == nil || p == localpkg || p->exported)
		return;
	p->exported = 1;
	suffix = "";
	if(!p->direct)
		suffix = " // indirect";
	Bprint(bout, "\timport %s \"%Z\"%s\n", p->name, p->path, suffix);
}

static void
dumpexportconst(Sym *s)
{
	Node *n;
	Type *t;

	n = s->def;
	typecheck(&n, Erv);
	if(n == N || n->op != OLITERAL)
		fatal("dumpexportconst: oconst nil: %S", s);

	t = n->type;	// may or may not be specified
	dumpexporttype(t);

	if(t != T && !isideal(t))
		Bprint(bout, "\tconst %#S %#T = %#V\n", s, t, &n->val);
	else
		Bprint(bout, "\tconst %#S = %#V\n", s, &n->val);
}

static void
dumpexportvar(Sym *s)
{
	Node *n;
	Type *t;

	n = s->def;
	typecheck(&n, Erv);
	if(n == N || n->type == T) {
		yyerror("variable exported but not defined: %S", s);
		return;
	}

	t = n->type;
	dumpexporttype(t);

	if(t->etype == TFUNC && n->class == PFUNC)
		Bprint(bout, "\tfunc %#S%#hT\n", s, t);
	else
		Bprint(bout, "\tvar %#S %#T\n", s, t);
}

static int
methcmp(const void *va, const void *vb)
{
	Type *a, *b;
	
	a = *(Type**)va;
	b = *(Type**)vb;
	return strcmp(a->sym->name, b->sym->name);
}

static void
dumpexporttype(Type *t)
{
	Type *f;
	Type **m;
	int i, n;

	if(t == T)
		return;

	if(t->printed || t == types[t->etype] || t == bytetype || t == runetype || t == errortype)
		return;
	t->printed = 1;

	if(t->sym != S && t->etype != TFIELD)
		dumppkg(t->sym->pkg);

	dumpexporttype(t->type);
	dumpexporttype(t->down);

	if (t->sym == S || t->etype == TFIELD)
		return;

	n = 0;
	for(f=t->method; f!=T; f=f->down) {	
		dumpexporttype(f);
		n++;
	}

	m = mal(n*sizeof m[0]);
	i = 0;
	for(f=t->method; f!=T; f=f->down)
		m[i++] = f;
	qsort(m, n, sizeof m[0], methcmp);

	Bprint(bout, "\ttype %#S %#lT\n", t->sym, t);
	for(i=0; i<n; i++) {
		f = m[i];
		Bprint(bout, "\tfunc (%#T) %#hS%#hT\n", getthisx(f->type)->type, f->sym, f->type);
	}
}

static void
dumpsym(Sym *s)
{
	if(s->flags & SymExported)
		return;
	s->flags |= SymExported;

	if(s->def == N) {
		yyerror("unknown export symbol: %S", s);
		return;
	}
	
	dumppkg(s->pkg);

	switch(s->def->op) {
	default:
		yyerror("unexpected export symbol: %O %S", s->def->op, s);
		break;
	case OLITERAL:
		dumpexportconst(s);
		break;
	case OTYPE:
		if(s->def->type->etype == TFORW)
			yyerror("export of incomplete type %S", s);
		else
			dumpexporttype(s->def->type);
		break;
	case ONAME:
		dumpexportvar(s);
		break;
	}
}

void
dumpexport(void)
{
	NodeList *l;
	int32 i, lno;
	Pkg *p;

	lno = lineno;

	Bprint(bout, "\n$$  // exports\n    package %s", localpkg->name);
	if(safemode)
		Bprint(bout, " safe");
	Bprint(bout, "\n");

	for(i=0; i<nelem(phash); i++)
		for(p=phash[i]; p; p=p->link)
			if(p->direct)
				dumppkg(p);

	for(l=exportlist; l; l=l->next) {
		lineno = l->n->lineno;
		dumpsym(l->n->sym);
	}

	Bprint(bout, "\n$$  // local types\n\n$$\n");   // 6l expects this. (see ld/go.c)

	lineno = lno;
}

/*
 * import
 */

/*
 * return the sym for ss, which should match lexical
 */
Sym*
importsym(Sym *s, int op)
{
	if(s->def != N && s->def->op != op)
		redeclare(s, "during import");

	// mark the symbol so it is not reexported
	if(s->def == N) {
		if(exportname(s->name) || initname(s->name))
			s->flags |= SymExport;
		else
			s->flags |= SymPackage;	// package scope
	}
	return s;
}

/*
 * return the type pkg.name, forward declaring if needed
 */
Type*
pkgtype(Sym *s)
{
	Type *t;

	importsym(s, OTYPE);
	if(s->def == N || s->def->op != OTYPE) {
		t = typ(TFORW);
		t->sym = s;
		s->def = typenod(t);
	}
	if(s->def->type == T)
		yyerror("pkgtype %S", s);
	return s->def->type;
}

static int
mypackage(Sym *s)
{
	// we import all definitions for runtime.
	// lowercase ones can only be used by the compiler.
	return s->pkg == localpkg || s->pkg == runtimepkg;
}

void
importconst(Sym *s, Type *t, Node *n)
{
	Node *n1;

	if(!exportname(s->name) && !mypackage(s))
		return;
	importsym(s, OLITERAL);
	convlit(&n, t);
	if(s->def != N) {
		// TODO: check if already the same.
		return;
	}

	if(n->op != OLITERAL) {
		yyerror("expression must be a constant");
		return;
	}
	if(n->sym != S) {
		n1 = nod(OXXX, N, N);
		*n1 = *n;
		n = n1;
	}
	n->sym = s;
	declare(n, PEXTERN);

	if(debug['E'])
		print("import const %S\n", s);
}

void
importvar(Sym *s, Type *t, int ctxt)
{
	Node *n;

	if(!exportname(s->name) && !initname(s->name) && !mypackage(s))
		return;

	importsym(s, ONAME);
	if(s->def != N && s->def->op == ONAME) {
		if(eqtype(t, s->def->type))
			return;
		yyerror("inconsistent definition for var %S during import\n\t%T\n\t%T", s, s->def->type, t);
	}
	n = newname(s);
	n->type = t;
	declare(n, ctxt);

	if(debug['E'])
		print("import var %S %lT\n", s, t);
}

void
importtype(Type *pt, Type *t)
{
	if(pt != T && t != T)
		typedcl2(pt, t);

	if(debug['E'])
		print("import type %T %lT\n", pt, t);
}

void
importmethod(Sym *s, Type *t)
{
	checkwidth(t);
	addmethod(s, t, 0);
}

