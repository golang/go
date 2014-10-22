// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Runtime type representation.

typedef struct Type Type;
typedef struct UncommonType UncommonType;
typedef struct InterfaceType InterfaceType;
typedef struct Method Method;
typedef struct IMethod IMethod;
typedef struct SliceType SliceType;
typedef struct FuncType FuncType;

// Needs to be in sync with ../../cmd/ld/decodesym.c:/^commonsize and pkg/reflect/type.go:/type.
struct Type
{
	uintptr size;
	uint32 hash;
	uint8 _unused;
	uint8 align;
	uint8 fieldAlign;
	uint8 kind;
	void* alg;
	// gc stores type info required for garbage collector.
	// If (kind&KindGCProg)==0, then gc[0] points at sparse GC bitmap
	// (no indirection), 4 bits per word.
	// If (kind&KindGCProg)!=0, then gc[1] points to a compiler-generated
	// read-only GC program; and gc[0] points to BSS space for sparse GC bitmap.
	// For huge types (>MaxGCMask), runtime unrolls the program directly into
	// GC bitmap and gc[0] is not used. For moderately-sized types, runtime
	// unrolls the program into gc[0] space on first use. The first byte of gc[0]
	// (gc[0][0]) contains 'unroll' flag saying whether the program is already
	// unrolled into gc[0] or not.
	uintptr gc[2];
	String *string;
	UncommonType *x;
	Type *ptrto;
	byte *zero;  // ptr to the zero value for this type
};

struct Method
{
	String *name;
	String *pkgPath;
	Type	*mtyp;
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

struct IMethod
{
	String *name;
	String *pkgPath;
	Type *type;
};

struct InterfaceType
{
	Type  typ;
	Slice mhdr;
	IMethod m[];
};

struct MapType
{
	Type typ;
	Type *key;
	Type *elem;
	Type *bucket;		// internal type representing a hash bucket
	Type *hmap;		// internal type representing a Hmap
	uint8 keysize;		// size of key slot
	bool indirectkey;	// store ptr to key instead of key itself
	uint8 valuesize;	// size of value slot
	bool indirectvalue;	// store ptr to value instead of value itself
	uint16 bucketsize;	// size of bucket
};

struct ChanType
{
	Type typ;
	Type *elem;
	uintptr dir;
};

struct SliceType
{
	Type typ;
	Type *elem;
};

struct FuncType
{
	Type typ;
	bool dotdotdot;
	Slice in;
	Slice out;
};

struct PtrType
{
	Type typ;
	Type *elem;
};
