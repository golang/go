// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Parse stabs debug info.

#include "a.h"

int stabsdebug = 1;

// Hash table for type lookup by number.
Type *hash[1024];

// Look up type by number pair.
// TODO(rsc): Iant points out that n1 and n2 are always small and dense,
// so an array of arrays would be a better representation.
Type*
typebynum(uint n1, uint n2)
{
	uint h;
	Type *t;

	h = (n1*53+n2) % nelem(hash);
	for(t=hash[h]; t; t=t->next)
		if(t->n1 == n1 && t->n2 == n2)
			return t;
	t = emalloc(sizeof *t);
	t->next = hash[h];
	hash[h] = t;
	t->n1 = n1;
	t->n2 = n2;
	return t;
}

// Parse name and colon from *pp, leaving copy in *sp.
static int
parsename(char **pp, char **sp)
{
	char *p;
	char *s;

	p = *pp;
	while(*p != '\0' && *p != ':')
		p++;
	if(*p == '\0') {
		fprint(2, "parsename expected colon\n");
		return -1;
	}
	s = emalloc(p - *pp + 1);
	memmove(s, *pp, p - *pp);
	*sp = s;
	*pp = p+1;
	return 0;
}

// Parse single number from *pp.
static int
parsenum1(char **pp, vlong *np)
{
	char *p;

	p = *pp;
	if(*p != '-' && (*p < '0' || *p > '9')) {
		fprint(2, "parsenum expected minus or digit\n");
		return -1;
	}
	*np = strtoll(p, pp, 10);
	return 0;
}

// Parse type number - either single number or (n1, n2).
static int
parsetypenum(char **pp, vlong *n1p, vlong *n2p)
{
	char *p;

	p = *pp;
	if(*p == '(') {
		p++;
		if(parsenum1(&p, n1p) < 0)
			return -1;
		if(*p++ != ',') {
			if(stabsdebug)
				fprint(2, "parsetypenum expected comma\n");
			return -1;
		}
		if(parsenum1(&p, n2p) < 0)
			return -1;
		if(*p++ != ')') {
			if(stabsdebug)
				fprint(2, "parsetypenum expected right paren\n");
			return -1;
		}
		*pp = p;
		return 0;
	}

	if(parsenum1(&p, n1p) < 0)
		return -1;
	*n2p = 0;
	*pp = p;
	return 0;
}

// Integer types are represented in stabs as a "range"
// type with a lo and a hi value.  The lo and hi used to
// be lo and hi for the type, but there are now odd
// extensions for floating point and 64-bit numbers.
//
// Have to keep signs separate from values because
// Int64's lo is -0.
typedef struct Intrange Intrange;
struct Intrange
{
	int signlo;	// sign of lo
	vlong lo;
	int signhi;	// sign of hi
	vlong hi;
	int kind;
};

// NOTE(rsc): Iant says that these might be different depending
// on the gcc mode, though I haven't observed this yet.
Intrange intranges[] = {
	'+', 0, '+', 127, Int8,	// char
	'-', 128, '+', 127, Int8,	// signed char
	'+', 0, '+', 255, Uint8,
	'-', 32768, '+', 32767, Int16,
	'+', 0, '+', 65535, Uint16,
	'-', 2147483648LL, '+', 2147483647LL, Int32,
	'+', 0, '+', 4294967295LL, Uint32,

	// abnormal cases
	'-', 0, '+', 4294967295LL, Int64,
	'+', 0, '-', 1, Uint64,

	'+', 4, '+', 0, Float32,
	'+', 8, '+', 0, Float64,
	'+', 16, '+', 0, Void,
};

static int kindsize[] = {
	0,
	8,
	8,
	16,
	16,
	32,
	32,
	64,
	64,
};

// Parse a single type definition from *pp.
static Type*
parsedef(char **pp, char *name)
{
	char *p;
	Type *t, *tt;
	int i, signlo, signhi;
	vlong n1, n2, lo, hi;
	Field *f;
	Intrange *r;

	p = *pp;

	// reference to another type?
	if(isdigit(*p) || *p == '(') {
		if(parsetypenum(&p, &n1, &n2) < 0)
			return nil;
		t = typebynum(n1, n2);
		if(name && t->name == nil) {
			t->name = name;
			// save definitions of names beginning with $
			if(name[0] == '$' && !t->saved) {
				typ = erealloc(typ, (ntyp+1)*sizeof typ[0]);
				typ[ntyp] = t;
				ntyp++;
			}
		}

		// is there an =def suffix?
		if(*p == '=') {
			p++;
			tt = parsedef(&p, name);
			if(tt == nil)
				return nil;

			if(tt == t) {
				tt->kind = Void;
			} else {
				t->type = tt;
				t->kind = Typedef;
			}

			// assign given name, but do not record in typ.
			// assume the name came from a typedef
			// which will be recorded.
			if(name)
				tt->name = name;
		}

		*pp = p;
		return t;
	}

	// otherwise a type literal.  first letter identifies kind
	t = emalloc(sizeof *t);
	switch(*p) {
	default:
		*pp = "";
		return t;

	case '*':	// pointer
		p++;
		t->kind = Ptr;
		tt = parsedef(&p, nil);
		if(tt == nil)
			return nil;
		t->type = tt;
		break;

	case 'a':	// array
		p++;
		t->kind = Array;
		// index type
		tt = parsedef(&p, nil);
		if(tt == nil)
			return nil;
		t->size = tt->size;
		// element type
		tt = parsedef(&p, nil);
		if(tt == nil)
			return nil;
		t->type = tt;
		break;

	case 'e':	// enum type - record $names in con array.
		p++;
		for(;;) {
			if(*p == '\0')
				return nil;
			if(*p == ';') {
				p++;
				break;
			}
			if(parsename(&p, &name) < 0)
				return nil;
			if(parsenum1(&p, &n1) < 0)
				return nil;
			if(name[0] == '$') {
				con = erealloc(con, (ncon+1)*sizeof con[0]);
				name++;
				con[ncon].name = name;
				con[ncon].value = n1;
				ncon++;
			}
			if(*p != ',')
				return nil;
			p++;
		}
		break;

	case 'f':	// function
		p++;
		if(parsedef(&p, nil) == nil)
			return nil;
		break;

	case 'r':	// sub-range (used for integers)
		p++;
		if(parsedef(&p, nil) == nil)
			return nil;
		// usually, the return from parsedef == t, but not always.

		if(*p != ';' || *++p == ';') {
			if(stabsdebug)
				fprint(2, "range expected number: %s\n", p);
			return nil;
		}
		if(*p == '-') {
			signlo = '-';
			p++;
		} else
			signlo = '+';
		lo = strtoll(p, &p, 10);
		if(*p != ';' || *++p == ';') {
			if(stabsdebug)
				fprint(2, "range expected number: %s\n", p);
			return nil;
		}
		if(*p == '-') {
			signhi = '-';
			p++;
		} else
			signhi = '+';
		hi = strtoll(p, &p, 10);
		if(*p != ';') {
			if(stabsdebug)
				fprint(2, "range expected trailing semi: %s\n", p);
			return nil;
		}
		p++;
		t->size = hi+1;	// might be array size
		for(i=0; i<nelem(intranges); i++) {
			r = &intranges[i];
			if(r->signlo == signlo && r->signhi == signhi && r->lo == lo && r->hi == hi) {
				t->kind = r->kind;
				break;
			}
		}
		break;

	case 's':	// struct
	case 'u':	// union
		t->kind = Struct;
		if(*p == 'u')
			t->kind = Union;

		// assign given name, but do not record in typ.
		// assume the name came from a typedef
		// which will be recorded.
		if(name)
			t->name = name;
		p++;
		if(parsenum1(&p, &n1) < 0)
			return nil;
		t->size = n1;
		for(;;) {
			if(*p == '\0')
				return nil;
			if(*p == ';') {
				p++;
				break;
			}
			t->f = erealloc(t->f, (t->nf+1)*sizeof t->f[0]);
			f = &t->f[t->nf];
			if(parsename(&p, &f->name) < 0)
				return nil;
			f->type = parsedef(&p, nil);
			if(f->type == nil)
				return nil;
			if(*p != ',') {
				fprint(2, "expected comma after def of %s:\n%s\n", f->name, p);
				return nil;
			}
			p++;
			if(parsenum1(&p, &n1) < 0)
				return nil;
			f->offset = n1;
			if(*p != ',') {
				fprint(2, "expected comma after offset of %s:\n%s\n", f->name, p);
				return nil;
			}
			p++;
			if(parsenum1(&p, &n1) < 0)
				return nil;
			f->size = n1;
			if(*p != ';') {
				fprint(2, "expected semi after size of %s:\n%s\n", f->name, p);
				return nil;
			}

			// rewrite
			//	uint32 x : 8;
			// into
			//	uint8 x;
			// hooray for bitfields.
			while(Int16 <= f->type->kind && f->type->kind <= Uint64 && kindsize[f->type->kind] > f->size) {
				t = emalloc(sizeof *t);
				*t = *f->type;
				f->type = t;
				f->type->kind -= 2;
			}
			p++;
			t->nf++;
		}
		break;


	}
	*pp = p;
	return t;
}


// Parse a stab type in p, saving info in the type hash table
// and also in the list of recorded types if appropriate.
void
parsestabtype(char *p)
{
	char *p0, *name;

	p0 = p;

	// p is the quoted string output from gcc -gstabs on a .stabs line.
	//	name:t(1,2)
	//	name:t(1,2)=def
	if(parsename(&p, &name) < 0) {
	Bad:
		// Use fprint instead of sysfatal to avoid
		// sysfatal's internal buffer size limit.
		fprint(2, "cannot parse stabs type:\n%s\n(at %s)\n", p0, p);
		sysfatal("stabs parse");
	}
	if(*p != 't' && *p != 'T')
		goto Bad;
	p++;

	// parse the definition.
	if(name[0] == '\0')
		name = nil;
	if(parsedef(&p, name) == nil)
		goto Bad;
	if(*p != '\0')
		goto Bad;
}

