// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
 * Runtime type representation; master is type.go
 */

typedef struct CommonType CommonType;
typedef struct UncommonType UncommonType;
typedef struct InterfaceType InterfaceType;
typedef struct Method Method;
typedef struct IMethod IMethod;
typedef struct MapType MapType;
typedef struct ChanType ChanType;

struct CommonType
{
	uintptr size;
	uint32 hash;
	uint8 alg;
	uint8 align;
	uint8 fieldAlign;
	String *string;
	UncommonType *x;
};

struct Method
{
	uint32 hash;
	String *name;
	String *pkgPath;
	Type *typ;
	void (*ifn)(void);
	void (*tfn)(void);
};

struct UncommonType
{
	String *name;
	String *pkgPath;
	Slice mhdr;
	Method m[];
};

struct Type
{
	void *type;	// interface{} value
	void *ptr;
	CommonType;
};

struct IMethod
{
	uint32 hash;
	uint32 perm;
	String *name;
	String *pkgPath;
	Type *type;
};

struct InterfaceType
{
	Type;
	Slice mhdr;
	IMethod m[];
};

struct MapType
{
	Type;
	Type *key;
	Type *elem;
};

struct ChanType
{
	Type;
	Type *elem;
	uintptr dir;
};
