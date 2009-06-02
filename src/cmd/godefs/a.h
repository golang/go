// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <u.h>
#include <libc.h>
#include <bio.h>

enum
{
	Void = 1,
	Int8,
	Uint8,
	Int16,
	Uint16,
	Int32,
	Uint32,
	Int64,
	Uint64,
	Float32,
	Float64,
	Ptr,
	Struct,
	Array,
	Union,
	Typedef,
};

typedef struct Field Field;
typedef struct Type Type;

struct Type
{
	Type *next;	// next in hash table

	// stabs name and two-integer id
	char *name;
	int n1;
	int n2;

	// int kind
	int kind;

	// sub-type for ptr, array
	Type *type;

	// struct fields
	Field *f;
	int nf;
	int size;

	int saved;	// recorded in typ array
	int warned;	// warned about needing type
	int printed;	// has the definition been printed yet?
};

struct Field
{
	char *name;
	Type *type;
	int offset;
	int size;
};

// Constants
typedef struct Const Const;
struct Const
{
	char *name;
	vlong value;
};

// Recorded constants and types, to be printed.
extern Const *con;
extern int ncon;
extern Type **typ;
extern int ntyp;

// Language output
typedef struct Lang Lang;
struct Lang
{
	char *constbegin;
	char *constfmt;
	char *constend;

	char *typdef;

	char *structbegin;
	char *unionbegin;
	char *structpadfmt;
	char *structend;

	int (*typefmt)(Fmt*);
};

extern Lang go, c;

void*	emalloc(int);
char*	estrdup(char*);
void*	erealloc(void*, int);
void		parsestabtype(char*);
