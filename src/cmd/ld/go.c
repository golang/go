// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// go-specific code shared across loaders (5l, 6l, 8l).

#include	"l.h"
#include	"../ld/lib.h"

// accumulate all type information from .6 files.
// check for inconsistencies.

// TODO:
//	generate debugging section in binary.
//	once the dust settles, try to move some code to
//		libmach, so that other linkers and ar can share.

/*
 *	package import data
 */
typedef struct Import Import;
struct Import
{
	Import *hash;	// next in hash table
	char *prefix;	// "type", "var", "func", "const"
	char *name;
	char *def;
	char *file;
};
enum {
	NIHASH = 1024
};
static Import *ihash[NIHASH];
static int nimport;

static int
hashstr(char *name)
{
	int h;
	char *cp;

	h = 0;
	for(cp = name; *cp; h += *cp++)
		h *= 1119;
	if(h < 0)
		h = ~h;
	return h;
}

static Import *
ilookup(char *name)
{
	int h;
	Import *x;

	h = hashstr(name) % NIHASH;
	for(x=ihash[h]; x; x=x->hash)
		if(x->name[0] == name[0] && strcmp(x->name, name) == 0)
			return x;
	x = mal(sizeof *x);
	x->name = strdup(name);
	x->hash = ihash[h];
	ihash[h] = x;
	nimport++;
	return x;
}

static void loadpkgdata(char*, char*, int);
static void loaddynld(char*, char*, int);
static int parsemethod(char**, char*, char**);
static int parsepkgdata(char*, char**, char*, char**, char**, char**);

void
ldpkg(Biobuf *f, int64 len, char *filename)
{
	char *data, *p0, *p1;

	if(debug['g'])
		return;

	if((int)len != len) {
		fprint(2, "%s: too much pkg data in %s\n", argv0, filename);
		return;
	}
	data = mal(len+1);
	if(Bread(f, data, len) != len) {
		fprint(2, "%s: short pkg read %s\n", argv0, filename);
		return;
	}
	data[len] = '\0';

	// first \n$$ marks beginning of exports - skip rest of line
	p0 = strstr(data, "\n$$");
	if(p0 == nil)
		return;
	p0 += 3;
	while(*p0 != '\n' && *p0 != '\0')
		p0++;

	// second marks end of exports / beginning of local data
	p1 = strstr(p0, "\n$$");
	if(p1 == nil) {
		fprint(2, "%s: cannot find end of exports in %s\n", argv0, filename);
		return;
	}
	while(p0 < p1 && (*p0 == ' ' || *p0 == '\t' || *p0 == '\n'))
		p0++;
	if(p0 < p1) {
		if(strncmp(p0, "package ", 8) != 0) {
			fprint(2, "%s: bad package section in %s - %s\n", argv0, filename, p0);
			return;
		}
		p0 += 8;
		while(*p0 == ' ' || *p0 == '\t' || *p0 == '\n')
			p0++;
		while(*p0 != ' ' && *p0 != '\t' && *p0 != '\n')
			p0++;

		loadpkgdata(filename, p0, p1 - p0);
	}

	// local types begin where exports end.
	// skip rest of line after $$ we found above
	p0 = p1 + 3;
	while(*p0 != '\n' && *p0 != '\0')
		p0++;

	// local types end at next \n$$.
	p1 = strstr(p0, "\n$$");
	if(p1 == nil) {
		fprint(2, "%s: cannot find end of local types in %s\n", argv0, filename);
		return;
	}

	loadpkgdata(filename, p0, p1 - p0);

	// look for dynld section
	p0 = strstr(p1, "\n$$  // dynld");
	if(p0 != nil) {
		p0 = strchr(p0+1, '\n');
		if(p0 == nil) {
			fprint(2, "%s: found $$ // dynld but no newline in %s\n", argv0, filename);
			return;
		}
		p1 = strstr(p0, "\n$$");
		if(p1 == nil)
			p1 = strstr(p0, "\n!\n");
		if(p1 == nil) {
			fprint(2, "%s: cannot find end of // dynld section in %s\n", argv0, filename);
			return;
		}
		loaddynld(filename, p0 + 1, p1 - p0);
	}
}

/*
 * a and b don't match.
 * is one a forward declaration and the other a valid completion?
 * if so, return the one to keep.
 */
char*
forwardfix(char *a, char *b)
{
	char *t;

	if(strlen(a) > strlen(b)) {
		t = a;
		a = b;
		b = t;
	}
	if(strcmp(a, "struct") == 0 && strncmp(b, "struct ", 7) == 0)
		return b;
	if(strcmp(a, "interface") == 0 && strncmp(b, "interface ", 10) == 0)
		return b;
	return nil;
}

static void
loadpkgdata(char *file, char *data, int len)
{
	char *p, *ep, *prefix, *name, *def, *ndef;
	Import *x;

	file = strdup(file);
	p = data;
	ep = data + len;
	while(parsepkgdata(file, &p, ep, &prefix, &name, &def) > 0) {
		x = ilookup(name);
		if(x->prefix == nil) {
			x->prefix = prefix;
			x->def = def;
			x->file = file;
		} else if(strcmp(x->prefix, prefix) != 0) {
			fprint(2, "%s: conflicting definitions for %s\n", argv0, name);
			fprint(2, "%s:\t%s %s ...\n", x->file, x->prefix, name);
			fprint(2, "%s:\t%s %s ...\n", file, prefix, name);
			nerrors++;
		} else if(strcmp(x->def, def) == 0) {
			// fine
		} else if((ndef = forwardfix(x->def, def)) != nil) {
			x->def = ndef;
		} else {
			fprint(2, "%s: conflicting definitions for %s\n", argv0, name);
			fprint(2, "%s:\t%s %s %s\n", x->file, x->prefix, name, x->def);
			fprint(2, "%s:\t%s %s %s\n", file, prefix, name, def);
			nerrors++;
		}
	}
}

static int
parsepkgdata(char *file, char **pp, char *ep, char **prefixp, char **namep, char **defp)
{
	char *p, *prefix, *name, *def, *edef, *meth;
	int n;

	// skip white space
	p = *pp;
	while(p < ep && (*p == ' ' || *p == '\t' || *p == '\n'))
		p++;
	if(p == ep || strncmp(p, "$$\n", 3) == 0)
		return 0;

	// prefix: (var|type|func|const)
	prefix = p;
	if(p + 6 > ep)
		return -1;
	if(strncmp(p, "var ", 4) == 0)
		p += 4;
	else if(strncmp(p, "type ", 5) == 0)
		p += 5;
	else if(strncmp(p, "func ", 5) == 0)
		p += 5;
	else if(strncmp(p, "const ", 6) == 0)
		p += 6;
	else {
		fprint(2, "%s: confused in pkg data near <<%.40s>>\n", argv0, prefix);
		nerrors++;
		return -1;
	}
	p[-1] = '\0';

	// name: a.b followed by space
	name = p;
	while(p < ep && *p != ' ')
		p++;
	if(p >= ep)
		return -1;
	*p++ = '\0';

	// def: free form to new line
	def = p;
	while(p < ep && *p != '\n')
		p++;
	if(p >= ep)
		return -1;
	edef = p;
	*p++ = '\0';

	// include methods on successive lines in def of named type
	while(parsemethod(&p, ep, &meth) > 0) {
		*edef++ = '\n';	// overwrites '\0'
		if(edef+1 > meth) {
			// We want to indent methods with a single \t.
			// 6g puts at least one char of indent before all method defs,
			// so there will be room for the \t.  If the method def wasn't
			// indented we could do something more complicated,
			// but for now just diagnose the problem and assume
			// 6g will keep indenting for us.
			fprint(2, "%s: %s: expected methods to be indented %p %p %.10s\n", argv0,
				file, edef, meth, meth);
			nerrors++;
			return -1;
		}
		*edef++ = '\t';
		n = strlen(meth);
		memmove(edef, meth, n);
		edef += n;
	}

	// done
	*pp = p;
	*prefixp = prefix;
	*namep = name;
	*defp = def;
	return 1;
}

static int
parsemethod(char **pp, char *ep, char **methp)
{
	char *p;

	// skip white space
	p = *pp;
	while(p < ep && (*p == ' ' || *p == '\t'))
		p++;
	if(p == ep)
		return 0;

	// if it says "func (", it's a method
	if(p + 6 >= ep || strncmp(p, "func (", 6) != 0)
		return 0;

	// definition to end of line
	*methp = p;
	while(p < ep && *p != '\n')
		p++;
	if(p >= ep) {
		fprint(2, "%s: lost end of line in method definition\n", argv0);
		*pp = ep;
		return -1;
	}
	*p++ = '\0';
	*pp = p;
	return 1;
}

static void
loaddynld(char *file, char *p, int n)
{
	char *next, *name, *def, *p0, *lib;
	Sym *s;

	p[n] = '\0';

	p0 = p;
	for(; *p; p=next) {
		next = strchr(p, '\n');
		if(next == nil)
			next = "";
		else
			*next++ = '\0';
		p0 = p;
		if(strncmp(p, "dynld ", 6) != 0)
			goto err;
		p += 6;
		name = p;
		p = strchr(name, ' ');
		if(p == nil)
			goto err;
		while(*p == ' ')
			p++;
		def = p;
		p = strchr(def, ' ');
		if(p == nil)
			goto err;
		while(*p == ' ')
			p++;
		lib = p;

		// successful parse: now can edit the line
		*strchr(name, ' ') = 0;
		*strchr(def, ' ') = 0;

		s = lookup(name, 0);
		s->dynldlib = lib;
		s->dynldname = def;
	}
	return;

err:
	fprint(2, "%s: invalid dynld line: %s\n", argv0, p0);
	nerrors++;
}

static void mark(Sym*);
static int markdepth;

static void
markdata(Prog *p, Sym *s)
{
	markdepth++;
	if(p != P && debug['v'] > 1)
		Bprint(&bso, "%d markdata %s\n", markdepth, s->name);
	for(; p != P; p=p->dlink)
		if(p->to.sym)
			mark(p->to.sym);
	markdepth--;
}

static void
marktext(Prog *p)
{
	Auto *a;

	if(p == P)
		return;
	if(p->as != ATEXT) {
		diag("marktext: %P", p);
		return;
	}
	for(a=p->to.autom; a; a=a->link)
		mark(a->gotype);
	markdepth++;
	if(debug['v'] > 1)
		Bprint(&bso, "%d marktext %s\n", markdepth, p->from.sym->name);
	for(a=p->to.autom; a; a=a->link)
		mark(a->gotype);
	for(p=p->link; p != P; p=p->link) {
		if(p->as == ATEXT || p->as == ADATA || p->as == AGLOBL)
			break;
		if(p->from.sym)
			mark(p->from.sym);
		if(p->to.sym)
			mark(p->to.sym);
	}
	markdepth--;
}

static void
mark(Sym *s)
{
	if(s == S || s->reachable)
		return;
	s->reachable = 1;
	if(s->text)
		marktext(s->text);
	if(s->data)
		markdata(s->data, s);
	if(s->gotype)
		mark(s->gotype);
}

static void
sweeplist(Prog **first, Prog **last)
{
	int reachable;
	Prog *p, *q;

	reachable = 1;
	q = P;
	for(p=*first; p != P; p=p->link) {
		switch(p->as) {
		case ATEXT:
		case ADATA:
		case AGLOBL:
			reachable = p->from.sym->reachable;
		}
		if(reachable) {
			if(q == P)
				*first = p;
			else
				q->link = p;
			q = p;
		}
	}
	if(q == P)
		*first = P;
	else
		q->link = P;
	*last = q;
}

static char*
morename[] =
{
	"sys·morestack",
	"sys·morestackx",

	"sys·morestack00",
	"sys·morestack10",
	"sys·morestack01",
	"sys·morestack11",

	"sys·morestack8",
	"sys·morestack16",
	"sys·morestack24",
	"sys·morestack32",
	"sys·morestack40",
	"sys·morestack48",
};

void
deadcode(void)
{
	int i;

	if(debug['v'])
		Bprint(&bso, "%5.2f deadcode\n", cputime());

	mark(lookup(INITENTRY, 0));
	for(i=0; i<nelem(morename); i++)
		mark(lookup(morename[i], 0));

	// remove dead data
	sweeplist(&datap, &edatap);
}
