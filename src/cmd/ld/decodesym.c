// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include	"l.h"
#include	"lib.h"
#include	"../../runtime/typekind.h"

// Decoding the type.* symbols.	 This has to be in sync with
// ../../runtime/type.go, or more specificaly, with what
// ../gc/reflect.c stuffs in these.

static Reloc*
decode_reloc(LSym *s, int32 off)
{
	int i;

	for (i = 0; i < s->nr; i++)
		if (s->r[i].off == off)
			return s->r + i;
	return nil;
}

static LSym*
decode_reloc_sym(LSym *s, int32 off)
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

static int
commonsize(void)
{
	return 8*PtrSize + 8;
}

// Type.commonType.kind
uint8
decodetype_kind(LSym *s)
{
	return s->p[1*PtrSize + 7] & KindMask;	//  0x13 / 0x1f
}

// Type.commonType.kind
uint8
decodetype_noptr(LSym *s)
{
	return s->p[1*PtrSize + 7] & KindNoPointers;	//  0x13 / 0x1f
}

// Type.commonType.kind
uint8
decodetype_usegcprog(LSym *s)
{
	return s->p[1*PtrSize + 7] & KindGCProg;	//  0x13 / 0x1f
}

// Type.commonType.size
vlong
decodetype_size(LSym *s)
{
	return decode_inuxi(s->p, PtrSize);	 // 0x8 / 0x10
}

// Type.commonType.gc
LSym*
decodetype_gcprog(LSym *s)
{
	return decode_reloc_sym(s, 1*PtrSize + 8 + 2*PtrSize);
}

uint8*
decodetype_gcmask(LSym *s)
{
	LSym *mask;
	
	mask = decode_reloc_sym(s, 1*PtrSize + 8 + 1*PtrSize);
	return mask->p;
}

// Type.ArrayType.elem and Type.SliceType.Elem
LSym*
decodetype_arrayelem(LSym *s)
{
	return decode_reloc_sym(s, commonsize());	// 0x1c / 0x30
}

vlong
decodetype_arraylen(LSym *s)
{
	return decode_inuxi(s->p + commonsize()+2*PtrSize, PtrSize);
}

// Type.PtrType.elem
LSym*
decodetype_ptrelem(LSym *s)
{
	return decode_reloc_sym(s, commonsize());	// 0x1c / 0x30
}

// Type.MapType.key, elem
LSym*
decodetype_mapkey(LSym *s)
{
	return decode_reloc_sym(s, commonsize());	// 0x1c / 0x30
}

LSym*
decodetype_mapvalue(LSym *s)
{
	return decode_reloc_sym(s, commonsize()+PtrSize);	// 0x20 / 0x38
}

// Type.ChanType.elem
LSym*
decodetype_chanelem(LSym *s)
{
	return decode_reloc_sym(s, commonsize());	// 0x1c / 0x30
}

// Type.FuncType.dotdotdot
int
decodetype_funcdotdotdot(LSym *s)
{
	return s->p[commonsize()];
}

// Type.FuncType.in.len
int
decodetype_funcincount(LSym *s)
{
	return decode_inuxi(s->p + commonsize()+2*PtrSize, IntSize);
}

int
decodetype_funcoutcount(LSym *s)
{
	return decode_inuxi(s->p + commonsize()+3*PtrSize + 2*IntSize, IntSize);
}

LSym*
decodetype_funcintype(LSym *s, int i)
{
	Reloc *r;

	r = decode_reloc(s, commonsize() + PtrSize);
	if (r == nil)
		return nil;
	return decode_reloc_sym(r->sym, r->add + i * PtrSize);
}

LSym*
decodetype_funcouttype(LSym *s, int i)
{
	Reloc *r;

	r = decode_reloc(s, commonsize() + 2*PtrSize + 2*IntSize);
	if (r == nil)
		return nil;
	return decode_reloc_sym(r->sym, r->add + i * PtrSize);
}

// Type.StructType.fields.Slice::len
int
decodetype_structfieldcount(LSym *s)
{
	return decode_inuxi(s->p + commonsize() + PtrSize, IntSize);
}

static int
structfieldsize(void)
{
	return 5*PtrSize;
}

// Type.StructType.fields[]-> name, typ and offset.
char*
decodetype_structfieldname(LSym *s, int i)
{
	Reloc *r;

	// go.string."foo"  0x28 / 0x40
	s = decode_reloc_sym(s, commonsize() + PtrSize + 2*IntSize + i*structfieldsize());
	if (s == nil)			// embedded structs have a nil name.
		return nil;
	r = decode_reloc(s, 0);		// s has a pointer to the string data at offset 0
	if (r == nil)			// shouldn't happen.
		return nil;
	return (char*) r->sym->p + r->add;	// the c-string
}

LSym*
decodetype_structfieldtype(LSym *s, int i)
{
	return decode_reloc_sym(s, commonsize() + PtrSize + 2*IntSize + i*structfieldsize() + 2*PtrSize);
}

vlong
decodetype_structfieldoffs(LSym *s, int i)
{
	return decode_inuxi(s->p + commonsize() + PtrSize + 2*IntSize + i*structfieldsize() + 4*PtrSize, IntSize);
}

// InterfaceTYpe.methods.len
vlong
decodetype_ifacemethodcount(LSym *s)
{
	return decode_inuxi(s->p + commonsize() + PtrSize, IntSize);
}
