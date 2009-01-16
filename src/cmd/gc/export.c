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
		if(dcladj != exportsym)
			warn("uppercase missing export: %S", s);
		exportsym(s);
	} else {
		if(dcladj == exportsym) {
			warn("export missing uppercase: %S", s);
			exportsym(s);
		} else
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
	if(s->export == 1)
		Bprint(bout, "export ");
	else if(s->export == 2)
		Bprint(bout, "package ");
	Bprint(bout, "const %lS ", s);
	if(t != T)
		Bprint(bout, "%#T ", t);
	Bprint(bout, " = ");

	switch(n->val.ctype) {
	default:
		fatal("dumpexportconst: unknown ctype: %S", s);
	case CTINT:
	case CTSINT:
	case CTUINT:
		Bprint(bout, "%B\n", n->val.u.xval);
		break;
	case CTBOOL:
		Bprint(bout, "0x%llux\n", n->val.u.bval);
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
	if(s->export == 1)
		Bprint(bout, "export ");
	else if(s->export == 2)
		Bprint(bout, "package ");
	if(t->etype == TFUNC)
		Bprint(bout, "func ");
	else
		Bprint(bout, "var ");
	Bprint(bout, "%lS %#T\n", s, t);
}

void
dumpexporttype(Sym *s)
{
	dumpprereq(s->otype);
	Bprint(bout, "\t");
	if(s->export == 1)
		Bprint(bout, "export ");
	else if(s->export == 2)
		Bprint(bout, "package ");
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
	case LBASETYPE:
		// TODO(rsc): sort methods by name
		for(f=s->otype->method; f!=T; f=f->down)
			dumpprereq(f);

		dumpexporttype(s);
		for(f=s->otype->method; f!=T; f=f->down)
			Bprint(bout, "\tfunc (%#T) %hS %#hT\n",
				f->type->type->type, f->sym, f->type);
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
importsym(int export, Node *ss, int lexical)
{
	Sym *s;

	renamepkg(ss);

	if(ss->op != OIMPORT)
		fatal("importsym: oops1 %N", ss);

	s = pkgsym(ss->sym->name, ss->psym->name, lexical);
	/* TODO botch - need some diagnostic checking for the following assignment */
	s->opackage = ss->osym->name;
	if(export) {
		if(s->export != export && s->export != 0)
			yyerror("export/package mismatch: %S", s);
		s->export = export;
	}
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
	n->osym = n->psym;
	renamepkg(n);
	s = importsym(0, n, LATYPE);

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
	return strcmp(ss->psym->name, package) == 0;
}

void
importconst(int export, Node *ss, Type *t, Val *v)
{
	Node *n;
	Sym *s;

	export = exportname(ss->sym->name);
	if(export == 2 && !mypackage(ss))
		return;

	n = nod(OLITERAL, N, N);
	n->val = *v;
	n->type = t;

	s = importsym(export, ss, LACONST);
	if(s->oconst != N) {
		// TODO: check if already the same.
		return;
	}

// fake out export vs upper checks until transition is over
if(export == 1) dcladj = exportsym;

	dodclconst(newname(s), n);

dcladj = nil;
	if(debug['e'])
		print("import const %S\n", s);
}

void
importvar(int export, Node *ss, Type *t)
{
	Sym *s;

	if(export == 2 && !mypackage(ss))
		return;

	s = importsym(export, ss, LNAME);
	if(s->oname != N) {
		if(eqtype(t, s->oname->type, 0))
			return;
		warn("redeclare import var %S from %T to %T",
			s, s->oname->type, t);
	}
	checkwidth(t);
	addvar(newname(s), t, PEXTERN);
	s->export = export;

	if(debug['e'])
		print("import var %S %lT\n", s, t);
}

void
importtype(int export, Node *ss, Type *t)
{
	Sym *s;

	s = importsym(export, ss, LATYPE);
	if(s->otype != T) {
		if(eqtype(t, s->otype, 0))
			return;
		if(s->otype->etype != TFORW) {
			warn("redeclare import type %S from %T to %T",
				s, s->otype, t);
			s->otype = typ(0);
		}
	}
	if(s->otype == T)
		s->otype = typ(0);
	*s->otype = *t;
	s->otype->sym = s;
	checkwidth(s->otype);

	// If type name should not be visible to importers,
	// hide it by setting the lexical type to name.
	// This will make references in the ordinary program
	// (but not the import sections) look at s->oname,
	// which is nil, as for an undefined name.
	if(export == 0 || (export == 2 && !mypackage(ss)))
		s->lexical = LNAME;

	if(debug['e'])
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



void
renamepkg(Node *n)
{
	if(n->psym == pkgimportname)
		if(pkgmyname != S)
			n->psym = pkgmyname;
}
