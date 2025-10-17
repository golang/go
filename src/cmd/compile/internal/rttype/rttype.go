// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package rttype allows the compiler to share type information with
// the runtime. The shared type information is stored in
// internal/abi. This package translates those types from the host
// machine on which the compiler runs to the target machine on which
// the compiled program will run. In particular, this package handles
// layout differences between e.g. a 64 bit compiler and 32 bit
// target.
package rttype

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/objw"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"internal/abi"
	"reflect"
)

// The type structures shared with the runtime.
var Type *types.Type

var ArrayType *types.Type
var ChanType *types.Type
var FuncType *types.Type
var InterfaceType *types.Type
var MapType *types.Type
var PtrType *types.Type
var SliceType *types.Type
var StructType *types.Type

// Types that are parts of the types above.
var IMethod *types.Type
var Method *types.Type
var StructField *types.Type
var UncommonType *types.Type

// Type switches and asserts
var InterfaceSwitch *types.Type
var TypeAssert *types.Type

// Interface tables (itabs)
var ITab *types.Type

func Init() {
	// Note: this has to be called explicitly instead of being
	// an init function so it runs after the types package has
	// been properly initialized.
	Type = FromReflect(reflect.TypeFor[abi.Type]())
	ArrayType = FromReflect(reflect.TypeFor[abi.ArrayType]())
	ChanType = FromReflect(reflect.TypeFor[abi.ChanType]())
	FuncType = FromReflect(reflect.TypeFor[abi.FuncType]())
	InterfaceType = FromReflect(reflect.TypeFor[abi.InterfaceType]())
	MapType = FromReflect(reflect.TypeFor[abi.MapType]())
	PtrType = FromReflect(reflect.TypeFor[abi.PtrType]())
	SliceType = FromReflect(reflect.TypeFor[abi.SliceType]())
	StructType = FromReflect(reflect.TypeFor[abi.StructType]())

	IMethod = FromReflect(reflect.TypeFor[abi.Imethod]())
	Method = FromReflect(reflect.TypeFor[abi.Method]())
	StructField = FromReflect(reflect.TypeFor[abi.StructField]())
	UncommonType = FromReflect(reflect.TypeFor[abi.UncommonType]())

	InterfaceSwitch = FromReflect(reflect.TypeFor[abi.InterfaceSwitch]())
	TypeAssert = FromReflect(reflect.TypeFor[abi.TypeAssert]())

	ITab = FromReflect(reflect.TypeFor[abi.ITab]())

	// Make sure abi functions are correct. These functions are used
	// by the linker which doesn't have the ability to do type layout,
	// so we check the functions it uses here.
	ptrSize := types.PtrSize
	if got, want := int64(abi.CommonSize(ptrSize)), Type.Size(); got != want {
		base.Fatalf("abi.CommonSize() == %d, want %d", got, want)
	}
	if got, want := int64(abi.StructFieldSize(ptrSize)), StructField.Size(); got != want {
		base.Fatalf("abi.StructFieldSize() == %d, want %d", got, want)
	}
	if got, want := int64(abi.UncommonSize()), UncommonType.Size(); got != want {
		base.Fatalf("abi.UncommonSize() == %d, want %d", got, want)
	}
	if got, want := int64(abi.TFlagOff(ptrSize)), Type.OffsetOf("TFlag"); got != want {
		base.Fatalf("abi.TFlagOff() == %d, want %d", got, want)
	}
	if got, want := int64(abi.ITabTypeOff(ptrSize)), ITab.OffsetOf("Type"); got != want {
		base.Fatalf("abi.ITabTypeOff() == %d, want %d", got, want)
	}
}

// FromReflect translates from a host type to the equivalent target type.
func FromReflect(rt reflect.Type) *types.Type {
	t := reflectToType(rt)
	types.CalcSize(t)
	return t
}

// reflectToType converts from a reflect.Type (which is a compiler
// host type) to a *types.Type, which is a target type.  The result
// must be CalcSize'd before using.
func reflectToType(rt reflect.Type) *types.Type {
	switch rt.Kind() {
	case reflect.Bool:
		return types.Types[types.TBOOL]
	case reflect.Int:
		return types.Types[types.TINT]
	case reflect.Int8:
		return types.Types[types.TINT8]
	case reflect.Int16:
		return types.Types[types.TINT16]
	case reflect.Int32:
		return types.Types[types.TINT32]
	case reflect.Uint8:
		return types.Types[types.TUINT8]
	case reflect.Uint16:
		return types.Types[types.TUINT16]
	case reflect.Uint32:
		return types.Types[types.TUINT32]
	case reflect.Float32:
		return types.Types[types.TFLOAT32]
	case reflect.Float64:
		return types.Types[types.TFLOAT64]
	case reflect.Uintptr:
		return types.Types[types.TUINTPTR]
	case reflect.Ptr:
		return types.NewPtr(reflectToType(rt.Elem()))
	case reflect.Func, reflect.UnsafePointer:
		// TODO: there's no mechanism to distinguish different pointer types,
		// so we treat them all as unsafe.Pointer.
		return types.Types[types.TUNSAFEPTR]
	case reflect.Slice:
		return types.NewSlice(reflectToType(rt.Elem()))
	case reflect.Array:
		return types.NewArray(reflectToType(rt.Elem()), int64(rt.Len()))
	case reflect.Struct:
		fields := make([]*types.Field, rt.NumField())
		for i := 0; i < rt.NumField(); i++ {
			f := rt.Field(i)
			ft := reflectToType(f.Type)
			fields[i] = &types.Field{Sym: &types.Sym{Name: f.Name}, Type: ft}
		}
		return types.NewStruct(fields)
	case reflect.Chan:
		return types.NewChan(reflectToType(rt.Elem()), types.ChanDir(rt.ChanDir()))
	case reflect.String:
		return types.Types[types.TSTRING]
	case reflect.Complex128:
		return types.Types[types.TCOMPLEX128]
	default:
		base.Fatalf("unhandled kind %s", rt.Kind())
		return nil
	}
}

// A Cursor represents a typed location inside a static variable where we
// are going to write.
type Cursor struct {
	lsym   *obj.LSym
	offset int64
	typ    *types.Type
}

// NewCursor returns a cursor starting at lsym+off and having type t.
func NewCursor(lsym *obj.LSym, off int64, t *types.Type) Cursor {
	return Cursor{lsym: lsym, offset: off, typ: t}
}

// WritePtr writes a pointer "target" to the component at the location specified by c.
func (c Cursor) WritePtr(target *obj.LSym) {
	if c.typ.Kind() != types.TUNSAFEPTR && c.typ.Kind() != types.TPTR {
		base.Fatalf("can't write ptr, it has kind %s", c.typ.Kind())
	}
	if target == nil {
		objw.Uintptr(c.lsym, int(c.offset), 0)
	} else {
		objw.SymPtr(c.lsym, int(c.offset), target, 0)
	}
}
func (c Cursor) WritePtrWeak(target *obj.LSym) {
	if c.typ.Kind() != types.TUINTPTR {
		base.Fatalf("can't write ptr, it has kind %s", c.typ.Kind())
	}
	objw.SymPtrWeak(c.lsym, int(c.offset), target, 0)
}
func (c Cursor) WriteUintptr(val uint64) {
	if c.typ.Kind() != types.TUINTPTR {
		base.Fatalf("can't write uintptr, it has kind %s", c.typ.Kind())
	}
	objw.Uintptr(c.lsym, int(c.offset), val)
}
func (c Cursor) WriteUint32(val uint32) {
	if c.typ.Kind() != types.TUINT32 {
		base.Fatalf("can't write uint32, it has kind %s", c.typ.Kind())
	}
	objw.Uint32(c.lsym, int(c.offset), val)
}
func (c Cursor) WriteUint16(val uint16) {
	if c.typ.Kind() != types.TUINT16 {
		base.Fatalf("can't write uint16, it has kind %s", c.typ.Kind())
	}
	objw.Uint16(c.lsym, int(c.offset), val)
}
func (c Cursor) WriteUint8(val uint8) {
	if c.typ.Kind() != types.TUINT8 {
		base.Fatalf("can't write uint8, it has kind %s", c.typ.Kind())
	}
	objw.Uint8(c.lsym, int(c.offset), val)
}
func (c Cursor) WriteInt(val int64) {
	if c.typ.Kind() != types.TINT {
		base.Fatalf("can't write int, it has kind %s", c.typ.Kind())
	}
	objw.Uintptr(c.lsym, int(c.offset), uint64(val))
}
func (c Cursor) WriteInt32(val int32) {
	if c.typ.Kind() != types.TINT32 {
		base.Fatalf("can't write int32, it has kind %s", c.typ.Kind())
	}
	objw.Uint32(c.lsym, int(c.offset), uint32(val))
}
func (c Cursor) WriteBool(val bool) {
	if c.typ.Kind() != types.TBOOL {
		base.Fatalf("can't write bool, it has kind %s", c.typ.Kind())
	}
	objw.Bool(c.lsym, int(c.offset), val)
}

// WriteSymPtrOff writes a "pointer" to the given symbol. The symbol
// is encoded as a uint32 offset from the start of the section.
func (c Cursor) WriteSymPtrOff(target *obj.LSym, weak bool) {
	if c.typ.Kind() != types.TINT32 && c.typ.Kind() != types.TUINT32 {
		base.Fatalf("can't write SymPtr, it has kind %s", c.typ.Kind())
	}
	if target == nil {
		objw.Uint32(c.lsym, int(c.offset), 0)
	} else if weak {
		objw.SymPtrWeakOff(c.lsym, int(c.offset), target)
	} else {
		objw.SymPtrOff(c.lsym, int(c.offset), target)
	}
}

// WriteSlice writes a slice header to c. The pointer is target+off, the len and cap fields are given.
func (c Cursor) WriteSlice(target *obj.LSym, off, len, cap int64) {
	if c.typ.Kind() != types.TSLICE {
		base.Fatalf("can't write slice, it has kind %s", c.typ.Kind())
	}
	objw.SymPtr(c.lsym, int(c.offset), target, int(off))
	objw.Uintptr(c.lsym, int(c.offset)+types.PtrSize, uint64(len))
	objw.Uintptr(c.lsym, int(c.offset)+2*types.PtrSize, uint64(cap))
	// TODO: ability to switch len&cap. Maybe not needed here, as every caller
	// passes the same thing for both?
	if len != cap {
		base.Fatalf("len != cap (%d != %d)", len, cap)
	}
}

// Reloc adds a relocation from the current cursor position.
// Reloc fills in Off and Siz fields. Caller should fill in the rest (Type, others).
func (c Cursor) Reloc(rel obj.Reloc) {
	rel.Off = int32(c.offset)
	rel.Siz = uint8(c.typ.Size())
	c.lsym.AddRel(base.Ctxt, rel)
}

// Field selects the field with the given name from the struct pointed to by c.
func (c Cursor) Field(name string) Cursor {
	if c.typ.Kind() != types.TSTRUCT {
		base.Fatalf("can't call Field on non-struct %v", c.typ)
	}
	for _, f := range c.typ.Fields() {
		if f.Sym.Name == name {
			return Cursor{lsym: c.lsym, offset: c.offset + f.Offset, typ: f.Type}
		}
	}
	base.Fatalf("couldn't find field %s in %v", name, c.typ)
	return Cursor{}
}

func (c Cursor) Elem(i int64) Cursor {
	if c.typ.Kind() != types.TARRAY {
		base.Fatalf("can't call Elem on non-array %v", c.typ)
	}
	if i < 0 || i >= c.typ.NumElem() {
		base.Fatalf("element access out of bounds [%d] in [0:%d]", i, c.typ.NumElem())
	}
	elem := c.typ.Elem()
	return Cursor{lsym: c.lsym, offset: c.offset + i*elem.Size(), typ: elem}
}

type ArrayCursor struct {
	c Cursor // cursor pointing at first element
	n int    // number of elements
}

// NewArrayCursor returns a cursor starting at lsym+off and having n copies of type t.
func NewArrayCursor(lsym *obj.LSym, off int64, t *types.Type, n int) ArrayCursor {
	return ArrayCursor{
		c: NewCursor(lsym, off, t),
		n: n,
	}
}

// Elem selects element i of the array pointed to by c.
func (a ArrayCursor) Elem(i int) Cursor {
	if i < 0 || i >= a.n {
		base.Fatalf("element index %d out of range [0:%d]", i, a.n)
	}
	return Cursor{lsym: a.c.lsym, offset: a.c.offset + int64(i)*a.c.typ.Size(), typ: a.c.typ}
}

// ModifyArray converts a cursor pointing at a type [k]T to a cursor pointing
// at a type [n]T.
// Also returns the size delta, aka (n-k)*sizeof(T).
func (c Cursor) ModifyArray(n int) (ArrayCursor, int64) {
	if c.typ.Kind() != types.TARRAY {
		base.Fatalf("can't call ModifyArray on non-array %v", c.typ)
	}
	k := c.typ.NumElem()
	return ArrayCursor{c: Cursor{lsym: c.lsym, offset: c.offset, typ: c.typ.Elem()}, n: n}, (int64(n) - k) * c.typ.Elem().Size()
}
