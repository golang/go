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
	if(s->export != 0)
		return;
	s->export = 1;

	addexportsym(s);
}


void
dumpprereq(Type *t)
{
	if(t == T)
		return;

	if(t->printed)
		return;
	t->printed = 1;

	if(t->sym != S && t->etype != TFIELD && t->sym->name[0] != '_')
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
	if(s->export != 0)
		Bprint(bout, "export ");
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
		Bprint(bout, "%.17e\n", mpgetflt(n->val.u.fval));
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
	if(s->export != 0)
		Bprint(bout, "export ");
	if(t->etype == TFUNC)
		Bprint(bout, "func ");
	else
		Bprint(bout, "var ");
	Bprint(bout, "%lS %#T\n", s, t);
}

void
dumpexporttype(Sym *s)
{
	Bprint(bout, "\t");
	if(s->export != 0)
		Bprint(bout, "export ");
	Bprint(bout, "type %lS %l#T\n",  s, s->otype);
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
		dumpexporttype(s);
		for(f=s->otype->method; f!=T; f=f->down) {
			dumpprereq(f);
			Bprint(bout, "\tfunc (%#T) %hS %#T\n",
				f->type->type->type, f->sym, f->type);
		}
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
	int32 lno;
	char *pkg;

	exporting = 1;
	lno = lineno;

	Bprint(bout, "   import\n");
	Bprint(bout, "   $$\n");

	Bprint(bout, "    package %s\n", package);
	pkg = package;
	package = "$nopkg";

	for(d=exportlist->forw; d!=D; d=d->forw) {
		lineno = d->lineno;
		dumpsym(d->dsym);
	}

	package = pkg;

	Bprint(bout, "\n$$\n");

	lineno = lno;
	exporting = 0;
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

	renamepkg(ss);

	if(ss->op != OIMPORT)
		fatal("importsym: oops1 %N", ss);

	s = pkgsym(ss->sym->name, ss->psym->name, lexical);

	/* TODO botch - need some diagnostic checking for the following assignment */
	s->opackage = ss->osym->name;
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
	s = importsym(n, LATYPE);

	if(s->otype == T) {
		t = typ(TFORW);
		t->sym = s;
		s->otype = t;
	}
	return s->otype;
}

void
importconst(int export, Node *ss, Type *t, Val *v)
{
	Node *n;
	Sym *s;

	n = nod(OLITERAL, N, N);
	n->val = *v;
	n->type = t;

	s = importsym(ss, LNAME);
	if(s->oconst != N) {
		// TODO: check if already the same.
		return;
	}

	dodclconst(newname(s), n);

	if(debug['e'])
		print("import const %S\n", s);
}

void
importvar(int export, Node *ss, Type *t)
{
	Sym *s;

	s = importsym(ss, LNAME);
	if(s->oname != N) {
		if(eqtype(t, s->oname->type, 0))
			return;
		warn("redeclare import var %S from %T to %T",
			s, s->oname->type, t);
	}
	checkwidth(t);
	addvar(newname(s), t, PEXTERN);

	if(debug['e'])
		print("import var %S %lT\n", s, t);
}

void
importtype(int export, Node *ss, Type *t)
{
	Sym *s;

	s = importsym(ss, LATYPE);
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
