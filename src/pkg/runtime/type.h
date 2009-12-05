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
typedef struct SliceType SliceType;

struct CommonType
{
	uintptr size;
	uint32 hash;
	uint8 alg;
	uint8 align;
	uint8 fieldAlign;
	uint8 kind;
	String *string;
	UncommonType *x;
};

enum {
	KindBool = 1,
	KindInt,
	KindInt8,
	KindInt16,
	KindInt32,
	KindInt64,
	KindUint,
	KindUint8,
	KindUint16,
	KindUint32,
	KindUint64,
	KindUintptr,
	KindFloat,
	KindFloat32,
	KindFloat64,
	KindArray,
	KindChan,
	KindDotDotDot,
	KindFunc,
	KindInterface,
	KindMap,
	KindPtr,
	KindSlice,
	KindString,
	KindStruct,
	KindUnsafePointer,
	
	KindNoPointers = 1<<7,
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

struct SliceType
{
	Type;
	Type *elem;
};

