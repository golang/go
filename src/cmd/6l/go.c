// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// go-specific

// accumulate all type information from .6 files.
// check for inconsistencies.
// define gotypestrings variable if needed.
// define gotypesigs variable if needed.

// TODO:
//	include type info for non-exported types.
//	generate debugging section in binary.
//	once the dust settles, try to move some code to
//		libmach, so that other linkers and ar can share.
//	try to make this completely portable and shared
//		across linkers

#include "l.h"

/*
 *	package import data
 */
typedef struct Import Import;
struct Import
{
	Import *hash;	// next in hash table
	int export;	// marked as export?
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
	x->name = name;
	x->hash = ihash[h];
	ihash[h] = x;
	nimport++;
	return x;
}

char*
gotypefor(char *name)
{
	Import *x;
	char *s, *p;

	s = strdup(name);
	p = utfrune(s, 0xB7);	// center dot
	if(p == nil)
		return nil;
	*p++ = '.';
	memmove(p, p+1, strlen(p));
	x = ilookup(s);
	free(s);
	if(x == nil || x->prefix == nil)
		return nil;
	if(strcmp(x->prefix, "var") != 0 && strcmp(x->prefix, "func") != 0)
		return nil;
	return x->def;
}

static void loadpkgdata(char*, char*, int);
static int parsemethod(char**, char*, char**);
static int parsepkgdata(char*, char**, char*, int*, char**, char**, char**);

void
ldpkg(Biobuf *f, int64 len, char *filename)
{
	char *data, *p0, *p1;

	if(debug['g'])
		return;

	if((int)len != len) {
		fprint(2, "6l: too much pkg data in %s\n", filename);
		return;
	}
	data = mal(len);
	if(Bread(f, data, len) != len) {
		fprint(2, "6l: short pkg read %s\n", filename);
		return;
	}

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
		fprint(2, "6l: cannot find end of exports in %s\n", filename);
		return;
	}
	while(*p0 == ' ' || *p0 == '\t' || *p0 == '\n')
		p0++;
	if(strncmp(p0, "package ", 8) != 0) {
		fprint(2, "6l: bad package section in %s\n", filename);
		return;
	}
	p0 += 8;
	while(*p0 == ' ' || *p0 == '\t' || *p0 == '\n')
		p0++;
	while(*p0 != ' ' && *p0 != '\t' && *p0 != '\n')
		p0++;

	loadpkgdata(filename, p0, p1 - p0);

	// local types begin where exports end.
	// skip rest of line after $$ we found above
	p0 = p1 + 3;
	while(*p0 != '\n' && *p0 != '\0')
		p0++;

	// local types end at next \n$$.
	p1 = strstr(p0, "\n$$");
	if(p1 == nil) {
		fprint(2, "6l: cannot find end of local types in %s\n", filename);
		return;
	}

	loadpkgdata(filename, p0, p1 - p0);
}

static void
loadpkgdata(char *file, char *data, int len)
{
	int export;
	char *p, *ep, *prefix, *name, *def;
	Import *x;

	file = strdup(file);
	p = data;
	ep = data + len;
	while(parsepkgdata(file, &p, ep, &export, &prefix, &name, &def) > 0) {
		x = ilookup(name);
		if(x->prefix == nil) {
			x->prefix = prefix;
			x->def = def;
			x->file = file;
			x->export = export;
		} else {
			if(strcmp(x->prefix, prefix) != 0) {
				fprint(2, "6l: conflicting definitions for %s\n", name);
				fprint(2, "%s:\t%s %s ...\n", x->file, x->prefix, name);
				fprint(2, "%s:\t%s %s ...\n", file, prefix, name);
				nerrors++;
			}
			else if(strcmp(x->def, def) != 0) {
				fprint(2, "6l: conflicting definitions for %s\n", name);
				fprint(2, "%s:\t%s %s %s\n", x->file, x->prefix, name, x->def);
				fprint(2, "%s:\t%s %s %s\n", file, prefix, name, def);
				nerrors++;
			}

			// okay if some .6 say export and others don't.
			// all it takes is one.
			if(export)
				x->export = 1;
		}
	}
}

static int
parsepkgdata(char *file, char **pp, char *ep, int *exportp, char **prefixp, char **namep, char **defp)
{
	char *p, *prefix, *name, *def, *edef, *meth;
	int n;

	// skip white space
	p = *pp;
	while(p < ep && (*p == ' ' || *p == '\t' || *p == '\n'))
		p++;
	if(p == ep || strncmp(p, "$$\n", 3) == 0)
		return 0;

	// [export|package ]
	*exportp = 0;
	if(p + 7 <= ep && strncmp(p, "export ", 7) == 0) {
		*exportp = 1;
		p += 7;
	}
	else if(p + 8 <= ep && strncmp(p, "package ", 8) == 0) {
		*exportp = 2;
		p += 8;
	}

	// prefix: (var|type|func|const)
	prefix = p;

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
	else{
		fprint(2, "6l: confused in pkg data near <<%.20s>>\n", p);
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
			fprint(2, "6l: %s: expected methods to be indented %p %p %.10s\n",
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
		fprint(2, "6l: lost end of line in method definition\n");
		*pp = ep;
		return -1;
	}
	*p++ = '\0';
	*pp = p;
	return 1;
}

static int
importcmp(const void *va, const void *vb)
{
	Import *a, *b;

	a = *(Import**)va;
	b = *(Import**)vb;
	return strcmp(a->name, b->name);
}

static int
symcmp(const void *va, const void *vb)
{
	Sym *a, *b;

	a = *(Sym**)va;
	b = *(Sym**)vb;
	return strcmp(a->name, b->name);
}

// if there is an undefined reference to gotypestrings,
// create it.  c declaration is
//	extern char gotypestrings[];
// ironically, gotypestrings is a c variable, because there
// is no way to forward declare a string in go.
void
definetypestrings(void)
{
	int i, j, len, n, w;
	char *p;
	Import **all, *x;
	Fmt f;
	Prog *prog;
	Sym *s;

	if(debug['g'])
		return;

	if(debug['v'])
		Bprint(&bso, "%5.2f definetypestrings\n", cputime());

	s = lookup("gotypestrings", 0);
	if(s->type == 0)
		return;
	if(s->type != SXREF) {
		diag("gotypestrings already defined");
		return;
	}
	s->type = SDATA;

	// make a list of all the type exports
	n = 0;
	for(i=0; i<NIHASH; i++)
		for(x=ihash[i]; x; x=x->hash)
			if(strcmp(x->prefix, "type") == 0)
				n++;
	all = mal(n*sizeof all[0]);
	j = 0;
	for(i=0; i<NIHASH; i++)
		for(x=ihash[i]; x; x=x->hash)
			if(strcmp(x->prefix, "type") == 0)
				all[j++] = x;

	// sort them by name
	qsort(all, n, sizeof all[0], importcmp);

	// make a big go string containing all the types
	fmtstrinit(&f);
	fmtprint(&f, "xxxx");	// 4-byte length
	for(i=0; i<n; i++) {
		p = strchr(all[i]->def, '\n');
		if(p)
			len = p - all[i]->def;
		else
			len = strlen(all[i]->def);
		fmtprint(&f, "%s %.*s\n", all[i]->name, utfnlen(all[i]->def, len), all[i]->def);
	}
	p = fmtstrflush(&f);
	n = strlen(p);
	s->value = n;

	// go strings begin with 4-byte length.
	// amd64 is little-endian.
	len = n - 4;
	p[0] = len;
	p[1] = len >> 8;
	p[2] = len >> 16;
	p[3] = len >> 24;

	// have data, need to create linker representation.
	// linker stores big data as sequence of pieces
	// with int8 length, so break p into 100-byte chunks.
	// (had to add D_SBIG even to do that; the compiler
	// would have generated 8-byte chunks.)
	for(i=0; i<n; i+=100) {
		w = 100;
		if(w > n - i)
			w = n - i;
		prog = newdata(s, i, w, D_EXTERN);
		prog->to.type = D_SBIG;
		prog->to.sbig = p + i;
	}

	if(debug['v'])
		Bprint(&bso, "%5.2f typestrings %d\n", cputime(), n);
}

// if there is an undefined reference to gotypesigs, create it.
// c declaration is
//	extern Sigt *gotypesigs[];
//	extern int ngotypesigs;
// used by sys.unreflect runtime.
void
definetypesigs(void)
{
	int i, j, n;
	Sym **all, *s, *x;
	Prog *prog;

	if(debug['g'])
		return;

	if(debug['v'])
		Bprint(&bso, "%5.2f definetypesigs\n", cputime());

	s = lookup("gotypesigs", 0);
	if(s->type == 0)
		return;
	if(s->type != SXREF) {
		diag("gotypesigs already defined");
		return;
	}
	s->type = SDATA;

	// make a list of all the sigt symbols.
	n = 0;
	for(i=0; i<NHASH; i++)
		for(x = hash[i]; x; x=x->link)
			if(memcmp(x->name, "sigt·", 6) == 0)
				n++;
	all = mal(n*sizeof all[0]);
	j = 0;
	for(i=0; i<NHASH; i++)
		for(x = hash[i]; x; x=x->link)
			if(memcmp(x->name, "sigt·", 6) == 0)
				all[j++] = x;

	// sort them by name
	qsort(all, n, sizeof all[0], symcmp);

	// emit array as sequence of references.
	enum { PtrSize = 8 };
	for(i=0; i<n; i++) {
		prog = newdata(s, PtrSize*i, PtrSize, D_EXTERN);
		prog->to.type = D_ADDR;
		prog->to.index = D_EXTERN;
		prog->to.sym = all[i];
	}
	s->value = PtrSize*n;
	if(n == 0)
		s->value = 1;	// must have non-zero size or 6l complains

	// emit count
	s = lookup("ngotypesigs", 0);
	s->type = SDATA;
	s->value = sizeof(int32);
	prog = newdata(s, 0, sizeof(int32), D_EXTERN);
	prog->to.offset = n;

	if(debug['v'])
		Bprint(&bso, "%5.2f typesigs %d\n", cputime(), n);

}
