// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include	"l.h"
#include	"lib.h"
#include	"../../pkg/runtime/typekind.h"

// Decoding the type.* symbols.	 This has to be in sync with
// ../../pkg/runtime/type.go, or more specificaly, with what
// ../gc/reflect.c stuffs in these.

static Reloc*
decode_reloc(Sym *s, int32 off)
{
	int i;

	for (i = 0; i < s->nr; i++)
		if (s->r[i].off == off)
			return s->r + i;
	return nil;
}

static Sym*
decode_reloc_sym(Sym *s, int32 off)
{
	Reloc *r;

	r = decode_reloc(s,off);
	if (r == nil)
		return nil;
	return r->sym;
}

static uvlong
decode_inuxi(uchar* p, int sz)
{
	uint64 v;
	uint32 l;
	uchar *cast, *inuxi;
	int i;

	v = l = 0;
	cast = nil;
	inuxi = nil;
	switch (sz) {
	case 2:
		cast = (uchar*)&l;
		inuxi = inuxi2;
		break;
	case 4:
		cast = (uchar*)&l;
		inuxi = inuxi4;
		break;
	case 8:
		cast = (uchar*)&v;
		inuxi = inuxi8;
		break;
	default:
		diag("dwarf: decode inuxi %d", sz);
		errorexit();
	}
	for (i = 0; i < sz; i++)
		cast[inuxi[i]] = p[i];
	if (sz == 8)
		return v;
	return l;
}

// Type.commonType.kind
uint8
decodetype_kind(Sym *s)
{
	return s->p[1*PtrSize + 7] & ~KindNoPointers;	//  0x13 / 0x1f
}

// Type.commonType.size
vlong
decodetype_size(Sym *s)
{
	return decode_inuxi(s->p, PtrSize);	 // 0x8 / 0x10
}

// Type.commonType.gc
Sym*
decodetype_gc(Sym *s)
{
	return decode_reloc_sym(s, 1*PtrSize + 8 + 1*PtrSize);
}

// Type.ArrayType.elem and Type.SliceType.Elem
Sym*
decodetype_arrayelem(Sym *s)
{
	return decode_reloc_sym(s, CommonSize);	// 0x1c / 0x30
}

vlong
decodetype_arraylen(Sym *s)
{
	return decode_inuxi(s->p + CommonSize+PtrSize, PtrSize);
}

// Type.PtrType.elem
Sym*
decodetype_ptrelem(Sym *s)
{
	return decode_reloc_sym(s, CommonSize);	// 0x1c / 0x30
}

// Type.MapType.key, elem
Sym*
decodetype_mapkey(Sym *s)
{
	return decode_reloc_sym(s, CommonSize);	// 0x1c / 0x30
}
Sym*
decodetype_mapvalue(Sym *s)
{
	return decode_reloc_sym(s, CommonSize+PtrSize);	// 0x20 / 0x38
}

// Type.ChanType.elem
Sym*
decodetype_chanelem(Sym *s)
{
	return decode_reloc_sym(s, CommonSize);	// 0x1c / 0x30
}

// Type.FuncType.dotdotdot
int
decodetype_funcdotdotdot(Sym *s)
{
	return s->p[CommonSize];
}

// Type.FuncType.in.len
int
decodetype_funcincount(Sym *s)
{
	return decode_inuxi(s->p + CommonSize+2*PtrSize, IntSize);
}

int
decodetype_funcoutcount(Sym *s)
{
	return decode_inuxi(s->p + CommonSize+3*PtrSize + 2*IntSize, IntSize);
}

Sym*
decodetype_funcintype(Sym *s, int i)
{
	Reloc *r;

	r = decode_reloc(s, CommonSize + PtrSize);
	if (r == nil)
		return nil;
	return decode_reloc_sym(r->sym, r->add + i * PtrSize);
}

Sym*
decodetype_funcouttype(Sym *s, int i)
{
	Reloc *r;

	r = decode_reloc(s, CommonSize + 2*PtrSize + 2*IntSize);
	if (r == nil)
		return nil;
	return decode_reloc_sym(r->sym, r->add + i * PtrSize);
}

// Type.StructType.fields.Slice::len
int
decodetype_structfieldcount(Sym *s)
{
	return decode_inuxi(s->p + CommonSize + PtrSize, IntSize);
}

enum {
	StructFieldSize = 5*PtrSize
};
// Type.StructType.fields[]-> name, typ and offset.
char*
decodetype_structfieldname(Sym *s, int i)
{
	Reloc *r;

	// go.string."foo"  0x28 / 0x40
	s = decode_reloc_sym(s, CommonSize + PtrSize + 2*IntSize + i*StructFieldSize);
	if (s == nil)			// embedded structs have a nil name.
		return nil;
	r = decode_reloc(s, 0);		// s has a pointer to the string data at offset 0
	if (r == nil)			// shouldn't happen.
		return nil;
	return (char*) r->sym->p + r->add;	// the c-string
}

Sym*
decodetype_structfieldtype(Sym *s, int i)
{
	return decode_reloc_sym(s, CommonSize + PtrSize + 2*IntSize + i*StructFieldSize + 2*PtrSize);
}

vlong
decodetype_structfieldoffs(Sym *s, int i)
{
	return decode_inuxi(s->p + CommonSize + PtrSize + 2*IntSize + i*StructFieldSize + 4*PtrSize, IntSize);
}

// InterfaceTYpe.methods.len
vlong
decodetype_ifacemethodcount(Sym *s)
{
	return decode_inuxi(s->p + CommonSize + PtrSize, IntSize);
}
