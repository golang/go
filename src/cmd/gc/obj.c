// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <u.h>
#include <libc.h>
#include "go.h"

/*
 * architecture-independent object file output
 */

static	void	dumpglobls(void);

void
dumpobj(void)
{
	NodeList *externs, *tmp;

	bout = Bopen(outfile, OWRITE);
	if(bout == nil) {
		flusherrors();
		print("can't create %s: %r\n", outfile);
		errorexit();
	}

	Bprint(bout, "go object %s %s %s %s\n", getgoos(), thestring, getgoversion(), expstring());
	Bprint(bout, "  exports automatically generated from\n");
	Bprint(bout, "  %s in package \"%s\"\n", curio.infile, localpkg->name);
	dumpexport();
	Bprint(bout, "\n!\n");

	linkouthist(ctxt, bout);

	externs = nil;
	if(externdcl != nil)
		externs = externdcl->end;

	dumpglobls();
	dumptypestructs();

	// Dump extra globals.
	tmp = externdcl;
	if(externs != nil)
		externdcl = externs->next;
	dumpglobls();
	externdcl = tmp;

	dumpdata();
	linkwritefuncs(ctxt, bout);

	Bterm(bout);
}

static void
dumpglobls(void)
{
	Node *n;
	NodeList *l;

	// add globals
	for(l=externdcl; l; l=l->next) {
		n = l->n;
		if(n->op != ONAME)
			continue;

		if(n->type == T)
			fatal("external %N nil type\n", n);
		if(n->class == PFUNC)
			continue;
		if(n->sym->pkg != localpkg)
			continue;
		dowidth(n->type);

		ggloblnod(n);
	}

	for(l=funcsyms; l; l=l->next) {
		n = l->n;
		dsymptr(n->sym, 0, n->sym->def->shortname->sym, 0);
		ggloblsym(n->sym, widthptr, 1, 1);
	}
}

void
Bputname(Biobuf *b, LSym *s)
{
	Bwrite(b, s->name, strlen(s->name)+1);
}

LSym*
linksym(Sym *s)
{
	char *p;

	if(s == nil)
		return nil;
	if(s->lsym != nil)
		return s->lsym;
	if(isblanksym(s))
		s->lsym = linklookup(ctxt, "_", 0);
	else {
		p = smprint("%s.%s", s->pkg->prefix, s->name);
		s->lsym = linklookup(ctxt, p, 0);
		free(p);
	}
	return s->lsym;	
}

int
duint8(Sym *s, int off, uint8 v)
{
	return duintxx(s, off, v, 1);
}

int
duint16(Sym *s, int off, uint16 v)
{
	return duintxx(s, off, v, 2);
}

int
duint32(Sym *s, int off, uint32 v)
{
	return duintxx(s, off, v, 4);
}

int
duint64(Sym *s, int off, uint64 v)
{
	return duintxx(s, off, v, 8);
}

int
duintptr(Sym *s, int off, uint64 v)
{
	return duintxx(s, off, v, widthptr);
}

Sym*
stringsym(char *s, int len)
{
	static int gen;
	Sym *sym;
	int off, n, m;
	struct {
		Strlit lit;
		char buf[110];
	} tmp;
	Pkg *pkg;

	if(len > 100) {
		// huge strings are made static to avoid long names
		snprint(namebuf, sizeof(namebuf), ".gostring.%d", ++gen);
		pkg = localpkg;
	} else {
		// small strings get named by their contents,
		// so that multiple modules using the same string
		// can share it.
		tmp.lit.len = len;
		memmove(tmp.lit.s, s, len);
		tmp.lit.s[len] = '\0';
		snprint(namebuf, sizeof(namebuf), "\"%Z\"", &tmp.lit);
		pkg = gostringpkg;
	}
	sym = pkglookup(namebuf, pkg);
	
	// SymUniq flag indicates that data is generated already
	if(sym->flags & SymUniq)
		return sym;
	sym->flags |= SymUniq;
	sym->def = newname(sym);

	off = 0;
	
	// string header
	off = dsymptr(sym, off, sym, widthptr+widthint);
	off = duintxx(sym, off, len, widthint);
	
	// string data
	for(n=0; n<len; n+=m) {
		m = 8;
		if(m > len-n)
			m = len-n;
		off = dsname(sym, off, s+n, m);
	}
	off = duint8(sym, off, 0);  // terminating NUL for runtime
	off = (off+widthptr-1)&~(widthptr-1);  // round to pointer alignment
	ggloblsym(sym, off, 1, 1);

	return sym;	
}
