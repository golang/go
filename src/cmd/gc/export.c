// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include	<u.h>
#include	<libc.h>
#include	"go.h"
#include	"y.tab.h"

static void	dumpexporttype(Type *t);

// Mark n's symbol as exported
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

	if(debug['E'])
		print("export symbol %S\n", n->sym);
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

// exportedsym reports whether a symbol will be visible
// to files that import our package.
static int
exportedsym(Sym *sym)
{
	// Builtins are visible everywhere.
	if(sym->pkg == builtinpkg || sym->origpkg == builtinpkg)
		return 1;

	return sym->pkg == localpkg && exportname(sym->name);
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
	// -A is for cmd/gc/mkbuiltin script, so export everything
	if(debug['A'] || exportname(n->sym->name) || initname(n->sym->name))
		exportsym(n);
}

static void
dumppkg(Pkg *p)
{
	char *suffix;

	if(p == nil || p == localpkg || p->exported || p == builtinpkg)
		return;
	p->exported = 1;
	suffix = "";
	if(!p->direct)
		suffix = " // indirect";
	Bprint(bout, "\timport %s \"%Z\"%s\n", p->name, p->path, suffix);
}

// Look for anything we need for the inline body
static void reexportdep(Node *n);
static void
reexportdeplist(NodeList *ll)
{
	for(; ll ;ll=ll->next)
		reexportdep(ll->n);
}

static void
reexportdep(Node *n)
{
	Type *t;

	if(!n)
		return;

	//print("reexportdep %+hN\n", n);
	switch(n->op) {
	case ONAME:
		switch(n->class&~PHEAP) {
		case PFUNC:
			// methods will be printed along with their type
			// nodes for T.Method expressions
			if(n->left && n->left->op == OTYPE)
				break;
			// nodes for method calls.
			if(!n->type || n->type->thistuple > 0)
				break;
			// fallthrough
		case PEXTERN:
			if(n->sym && !exportedsym(n->sym)) {
				if(debug['E'])
					print("reexport name %S\n", n->sym);
				exportlist = list(exportlist, n);
			}
		}
		break;

	case ODCL:
		// Local variables in the bodies need their type.
		t = n->left->type;
		if(t != types[t->etype] && t != idealbool && t != idealstring) {
			if(isptr[t->etype])
				t = t->type;
			if(t && t->sym && t->sym->def && !exportedsym(t->sym)) {
				if(debug['E'])
					print("reexport type %S from declaration\n", t->sym);
				exportlist = list(exportlist, t->sym->def);
			}
		}
		break;

	case OLITERAL:
		t = n->type;
		if(t != types[n->type->etype] && t != idealbool && t != idealstring) {
			if(isptr[t->etype])
				t = t->type;
			if(t && t->sym && t->sym->def && !exportedsym(t->sym)) {
				if(debug['E'])
					print("reexport literal type %S\n", t->sym);
				exportlist = list(exportlist, t->sym->def);
			}
		}
		// fallthrough
	case OTYPE:
		if(n->sym && !exportedsym(n->sym)) {
			if(debug['E'])
				print("reexport literal/type %S\n", n->sym);
			exportlist = list(exportlist, n);
		}
		break;

	// for operations that need a type when rendered, put the type on the export list.
	case OCONV:
	case OCONVIFACE:
	case OCONVNOP:
	case ORUNESTR:
	case OARRAYBYTESTR:
	case OARRAYRUNESTR:
	case OSTRARRAYBYTE:
	case OSTRARRAYRUNE:
	case ODOTTYPE:
	case ODOTTYPE2:
	case OSTRUCTLIT:
	case OARRAYLIT:
	case OPTRLIT:
	case OMAKEMAP:
	case OMAKESLICE:
	case OMAKECHAN:
		t = n->type;
		if(!t->sym && t->type)
			t = t->type;
		if(t && t->sym && t->sym->def && !exportedsym(t->sym)) {
			if(debug['E'])
				print("reexport type for expression %S\n", t->sym);
			exportlist = list(exportlist, t->sym->def);
		}
		break;
	}

	reexportdep(n->left);
	reexportdep(n->right);
	reexportdeplist(n->list);
	reexportdeplist(n->rlist);
	reexportdeplist(n->ninit);
	reexportdep(n->ntest);
	reexportdep(n->nincr);
	reexportdeplist(n->nbody);
	reexportdeplist(n->nelse);
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
	typecheck(&n, Erv|Ecall);
	if(n == N || n->type == T) {
		yyerror("variable exported but not defined: %S", s);
		return;
	}

	t = n->type;
	dumpexporttype(t);

	if(t->etype == TFUNC && n->class == PFUNC) {
		if (n->inl) {
			// when lazily typechecking inlined bodies, some re-exported ones may not have been typechecked yet.
			// currently that can leave unresolved ONONAMEs in import-dot-ed packages in the wrong package
			if(debug['l'] < 2)
				typecheckinl(n);
			// NOTE: The space after %#S here is necessary for ld's export data parser.
			Bprint(bout, "\tfunc %#S %#hT { %#H }\n", s, t, n->inl);
			reexportdeplist(n->inl);
		} else
			Bprint(bout, "\tfunc %#S %#hT\n", s, t);
	} else
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
		if(f->nointerface)
			Bprint(bout, "\t//go:nointerface\n");
		if (f->type->nname && f->type->nname->inl) { // nname was set by caninl
			// when lazily typechecking inlined bodies, some re-exported ones may not have been typechecked yet.
			// currently that can leave unresolved ONONAMEs in import-dot-ed packages in the wrong package
			if(debug['l'] < 2)
				typecheckinl(f->type->nname);
			Bprint(bout, "\tfunc (%#T) %#hhS %#hT { %#H }\n", getthisx(f->type)->type, f->sym, f->type, f->type->nname->inl);
			reexportdeplist(f->type->nname->inl);
		} else
			Bprint(bout, "\tfunc (%#T) %#hhS %#hT\n", getthisx(f->type)->type, f->sym, f->type);
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
//	print("dumpsym %O %+S\n", s->def->op, s);
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

	Bprint(bout, "\n$$\npackage %s", localpkg->name);
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

	Bprint(bout, "\n$$\n");
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
	char *pkgstr;

	if(s->def != N && s->def->op != op) {
		pkgstr = smprint("during import \"%Z\"", importpkg->path);
		redeclare(s, pkgstr);
	}

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

void
importimport(Sym *s, Strlit *z)
{
	// Informational: record package name
	// associated with import path, for use in
	// human-readable messages.
	Pkg *p;

	if(isbadimport(z))
		errorexit();
	p = mkpkg(z);
	if(p->name == nil) {
		p->name = s->name;
		pkglookup(s->name, nil)->npkg++;
	} else if(strcmp(p->name, s->name) != 0)
		yyerror("conflicting names %s and %s for package \"%Z\"", p->name, s->name, p->path);
	
	if(!incannedimport && myimportpath != nil && strcmp(z->s, myimportpath) == 0) {
		yyerror("import \"%Z\": package depends on \"%Z\" (import cycle)", importpkg->path, z);
		errorexit();
	}
}

void
importconst(Sym *s, Type *t, Node *n)
{
	Node *n1;

	importsym(s, OLITERAL);
	convlit(&n, t);

	if(s->def != N)	 // TODO: check if already the same.
		return;

	if(n->op != OLITERAL) {
		yyerror("expression must be a constant");
		return;
	}

	if(n->sym != S) {
		n1 = nod(OXXX, N, N);
		*n1 = *n;
		n = n1;
	}
	n->orig = newname(s);
	n->sym = s;
	declare(n, PEXTERN);

	if(debug['E'])
		print("import const %S\n", s);
}

void
importvar(Sym *s, Type *t)
{
	Node *n;

	importsym(s, ONAME);
	if(s->def != N && s->def->op == ONAME) {
		if(eqtype(t, s->def->type))
			return;
		yyerror("inconsistent definition for var %S during import\n\t%T (in \"%Z\")\n\t%T (in \"%Z\")", s, s->def->type, s->importdef->path, t, importpkg->path);
	}
	n = newname(s);
	s->importdef = importpkg;
	n->type = t;
	declare(n, PEXTERN);

	if(debug['E'])
		print("import var %S %lT\n", s, t);
}

void
importtype(Type *pt, Type *t)
{
	Node *n;

	// override declaration in unsafe.go for Pointer.
	// there is no way in Go code to define unsafe.Pointer
	// so we have to supply it.
	if(incannedimport &&
	   strcmp(importpkg->name, "unsafe") == 0 &&
	   strcmp(pt->nod->sym->name, "Pointer") == 0) {
		t = types[TUNSAFEPTR];
	}

	if(pt->etype == TFORW) {
		n = pt->nod;
		copytype(pt->nod, t);
		pt->nod = n;		// unzero nod
		pt->sym->importdef = importpkg;
		pt->sym->lastlineno = parserline();
		declare(n, PEXTERN);
		checkwidth(pt);
	} else if(!eqtype(pt->orig, t))
		yyerror("inconsistent definition for type %S during import\n\t%lT (in \"%Z\")\n\t%lT (in \"%Z\")", pt->sym, pt, pt->sym->importdef->path, t, importpkg->path);

	if(debug['E'])
		print("import type %T %lT\n", pt, t);
}
