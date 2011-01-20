// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ogle

import (
	"debug/proc"
	"exp/eval"
	"fmt"
	"log"
)

const debugParseRemoteType = false

// A remoteType is the local representation of a type in a remote process.
type remoteType struct {
	eval.Type
	// The size of values of this type in bytes.
	size int
	// The field alignment of this type.  Only used for
	// manually-constructed types.
	fieldAlign int
	// The maker function to turn a remote address of a value of
	// this type into an interpreter Value.
	mk maker
}

var manualTypes = make(map[Arch]map[eval.Type]*remoteType)

// newManualType constructs a remote type from an interpreter Type
// using the size and alignment properties of the given architecture.
// Most types are parsed directly out of the remote process, but to do
// so we need to layout the structures that describe those types ourselves.
func newManualType(t eval.Type, arch Arch) *remoteType {
	if nt, ok := t.(*eval.NamedType); ok {
		t = nt.Def
	}

	// Get the type map for this architecture
	typeMap := manualTypes[arch]
	if typeMap == nil {
		typeMap = make(map[eval.Type]*remoteType)
		manualTypes[arch] = typeMap

		// Construct basic types for this architecture
		basicType := func(t eval.Type, mk maker, size int, fieldAlign int) {
			t = t.(*eval.NamedType).Def
			if fieldAlign == 0 {
				fieldAlign = size
			}
			typeMap[t] = &remoteType{t, size, fieldAlign, mk}
		}
		basicType(eval.Uint8Type, mkUint8, 1, 0)
		basicType(eval.Uint32Type, mkUint32, 4, 0)
		basicType(eval.UintptrType, mkUintptr, arch.PtrSize(), 0)
		basicType(eval.Int16Type, mkInt16, 2, 0)
		basicType(eval.Int32Type, mkInt32, 4, 0)
		basicType(eval.IntType, mkInt, arch.IntSize(), 0)
		basicType(eval.StringType, mkString, arch.PtrSize()+arch.IntSize(), arch.PtrSize())
	}

	if rt, ok := typeMap[t]; ok {
		return rt
	}

	var rt *remoteType
	switch t := t.(type) {
	case *eval.PtrType:
		var elem *remoteType
		mk := func(r remote) eval.Value { return remotePtr{r, elem} }
		rt = &remoteType{t, arch.PtrSize(), arch.PtrSize(), mk}
		// Construct the element type after registering the
		// type to break cycles.
		typeMap[eval.Type(t)] = rt
		elem = newManualType(t.Elem, arch)

	case *eval.ArrayType:
		elem := newManualType(t.Elem, arch)
		mk := func(r remote) eval.Value { return remoteArray{r, t.Len, elem} }
		rt = &remoteType{t, elem.size * int(t.Len), elem.fieldAlign, mk}

	case *eval.SliceType:
		elem := newManualType(t.Elem, arch)
		mk := func(r remote) eval.Value { return remoteSlice{r, elem} }
		rt = &remoteType{t, arch.PtrSize() + 2*arch.IntSize(), arch.PtrSize(), mk}

	case *eval.StructType:
		layout := make([]remoteStructField, len(t.Elems))
		offset := 0
		fieldAlign := 0
		for i, f := range t.Elems {
			elem := newManualType(f.Type, arch)
			if fieldAlign == 0 {
				fieldAlign = elem.fieldAlign
			}
			offset = arch.Align(offset, elem.fieldAlign)
			layout[i].offset = offset
			layout[i].fieldType = elem
			offset += elem.size
		}
		mk := func(r remote) eval.Value { return remoteStruct{r, layout} }
		rt = &remoteType{t, offset, fieldAlign, mk}

	default:
		log.Panicf("cannot manually construct type %T", t)
	}

	typeMap[t] = rt
	return rt
}

var prtIndent = ""

// parseRemoteType parses a Type structure in a remote process to
// construct the corresponding interpreter type and remote type.
func parseRemoteType(a aborter, rs remoteStruct) *remoteType {
	addr := rs.addr().base
	p := rs.addr().p

	// We deal with circular types by discovering cycles at
	// NamedTypes.  If a type cycles back to something other than
	// a named type, we're guaranteed that there will be a named
	// type somewhere in that cycle.  Thus, we continue down,
	// re-parsing types until we reach the named type in the
	// cycle.  In order to still create one remoteType per remote
	// type, we insert an empty remoteType in the type map the
	// first time we encounter the type and re-use that structure
	// the second time we encounter it.

	rt, ok := p.types[addr]
	if ok && rt.Type != nil {
		return rt
	} else if !ok {
		rt = &remoteType{}
		p.types[addr] = rt
	}

	if debugParseRemoteType {
		sym := p.syms.SymByAddr(uint64(addr))
		name := "<unknown>"
		if sym != nil {
			name = sym.Name
		}
		log.Printf("%sParsing type at %#x (%s)", prtIndent, addr, name)
		prtIndent += " "
		defer func() { prtIndent = prtIndent[0 : len(prtIndent)-1] }()
	}

	// Get Type header
	itype := proc.Word(rs.field(p.f.Type.Typ).(remoteUint).aGet(a))
	typ := rs.field(p.f.Type.Ptr).(remotePtr).aGet(a).(remoteStruct)

	// Is this a named type?
	var nt *eval.NamedType
	uncommon := typ.field(p.f.CommonType.UncommonType).(remotePtr).aGet(a)
	if uncommon != nil {
		name := uncommon.(remoteStruct).field(p.f.UncommonType.Name).(remotePtr).aGet(a)
		if name != nil {
			// TODO(austin) Declare type in appropriate remote package
			nt = eval.NewNamedType(name.(remoteString).aGet(a))
			rt.Type = nt
		}
	}

	// Create type
	var t eval.Type
	var mk maker
	switch itype {
	case p.runtime.PBoolType:
		t = eval.BoolType
		mk = mkBool
	case p.runtime.PUint8Type:
		t = eval.Uint8Type
		mk = mkUint8
	case p.runtime.PUint16Type:
		t = eval.Uint16Type
		mk = mkUint16
	case p.runtime.PUint32Type:
		t = eval.Uint32Type
		mk = mkUint32
	case p.runtime.PUint64Type:
		t = eval.Uint64Type
		mk = mkUint64
	case p.runtime.PUintType:
		t = eval.UintType
		mk = mkUint
	case p.runtime.PUintptrType:
		t = eval.UintptrType
		mk = mkUintptr
	case p.runtime.PInt8Type:
		t = eval.Int8Type
		mk = mkInt8
	case p.runtime.PInt16Type:
		t = eval.Int16Type
		mk = mkInt16
	case p.runtime.PInt32Type:
		t = eval.Int32Type
		mk = mkInt32
	case p.runtime.PInt64Type:
		t = eval.Int64Type
		mk = mkInt64
	case p.runtime.PIntType:
		t = eval.IntType
		mk = mkInt
	case p.runtime.PFloat32Type:
		t = eval.Float32Type
		mk = mkFloat32
	case p.runtime.PFloat64Type:
		t = eval.Float64Type
		mk = mkFloat64
	case p.runtime.PStringType:
		t = eval.StringType
		mk = mkString

	case p.runtime.PArrayType:
		// Cast to an ArrayType
		typ := p.runtime.ArrayType.mk(typ.addr()).(remoteStruct)
		len := int64(typ.field(p.f.ArrayType.Len).(remoteUint).aGet(a))
		elem := parseRemoteType(a, typ.field(p.f.ArrayType.Elem).(remotePtr).aGet(a).(remoteStruct))
		t = eval.NewArrayType(len, elem.Type)
		mk = func(r remote) eval.Value { return remoteArray{r, len, elem} }

	case p.runtime.PStructType:
		// Cast to a StructType
		typ := p.runtime.StructType.mk(typ.addr()).(remoteStruct)
		fs := typ.field(p.f.StructType.Fields).(remoteSlice).aGet(a)

		fields := make([]eval.StructField, fs.Len)
		layout := make([]remoteStructField, fs.Len)
		for i := range fields {
			f := fs.Base.(remoteArray).elem(int64(i)).(remoteStruct)
			elemrs := f.field(p.f.StructField.Typ).(remotePtr).aGet(a).(remoteStruct)
			elem := parseRemoteType(a, elemrs)
			fields[i].Type = elem.Type
			name := f.field(p.f.StructField.Name).(remotePtr).aGet(a)
			if name == nil {
				fields[i].Anonymous = true
			} else {
				fields[i].Name = name.(remoteString).aGet(a)
			}
			layout[i].offset = int(f.field(p.f.StructField.Offset).(remoteUint).aGet(a))
			layout[i].fieldType = elem
		}

		t = eval.NewStructType(fields)
		mk = func(r remote) eval.Value { return remoteStruct{r, layout} }

	case p.runtime.PPtrType:
		// Cast to a PtrType
		typ := p.runtime.PtrType.mk(typ.addr()).(remoteStruct)
		elem := parseRemoteType(a, typ.field(p.f.PtrType.Elem).(remotePtr).aGet(a).(remoteStruct))
		t = eval.NewPtrType(elem.Type)
		mk = func(r remote) eval.Value { return remotePtr{r, elem} }

	case p.runtime.PSliceType:
		// Cast to a SliceType
		typ := p.runtime.SliceType.mk(typ.addr()).(remoteStruct)
		elem := parseRemoteType(a, typ.field(p.f.SliceType.Elem).(remotePtr).aGet(a).(remoteStruct))
		t = eval.NewSliceType(elem.Type)
		mk = func(r remote) eval.Value { return remoteSlice{r, elem} }

	case p.runtime.PMapType, p.runtime.PChanType, p.runtime.PFuncType, p.runtime.PInterfaceType, p.runtime.PUnsafePointerType, p.runtime.PDotDotDotType:
		// TODO(austin)
		t = eval.UintptrType
		mk = mkUintptr

	default:
		sym := p.syms.SymByAddr(uint64(itype))
		name := "<unknown symbol>"
		if sym != nil {
			name = sym.Name
		}
		err := fmt.Sprintf("runtime type at %#x has unexpected type %#x (%s)", addr, itype, name)
		a.Abort(FormatError(err))
	}

	// Fill in the remote type
	if nt != nil {
		nt.Complete(t)
	} else {
		rt.Type = t
	}
	rt.size = int(typ.field(p.f.CommonType.Size).(remoteUint).aGet(a))
	rt.mk = mk

	return rt
}
