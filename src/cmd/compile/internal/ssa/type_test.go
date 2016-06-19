// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// Stub implementation used for testing.
type TypeImpl struct {
	Size_   int64
	Align   int64
	Boolean bool
	Integer bool
	Signed  bool
	Float   bool
	Complex bool
	Ptr     bool
	string  bool
	slice   bool
	array   bool
	struct_ bool
	inter   bool
	Elem_   Type

	Name string
}

func (t *TypeImpl) Size() int64            { return t.Size_ }
func (t *TypeImpl) Alignment() int64       { return t.Align }
func (t *TypeImpl) IsBoolean() bool        { return t.Boolean }
func (t *TypeImpl) IsInteger() bool        { return t.Integer }
func (t *TypeImpl) IsSigned() bool         { return t.Signed }
func (t *TypeImpl) IsFloat() bool          { return t.Float }
func (t *TypeImpl) IsComplex() bool        { return t.Complex }
func (t *TypeImpl) IsPtrShaped() bool      { return t.Ptr }
func (t *TypeImpl) IsString() bool         { return t.string }
func (t *TypeImpl) IsSlice() bool          { return t.slice }
func (t *TypeImpl) IsArray() bool          { return t.array }
func (t *TypeImpl) IsStruct() bool         { return t.struct_ }
func (t *TypeImpl) IsInterface() bool      { return t.inter }
func (t *TypeImpl) IsMemory() bool         { return false }
func (t *TypeImpl) IsFlags() bool          { return false }
func (t *TypeImpl) IsVoid() bool           { return false }
func (t *TypeImpl) String() string         { return t.Name }
func (t *TypeImpl) SimpleString() string   { return t.Name }
func (t *TypeImpl) ElemType() Type         { return t.Elem_ }
func (t *TypeImpl) PtrTo() Type            { panic("not implemented") }
func (t *TypeImpl) NumFields() int         { panic("not implemented") }
func (t *TypeImpl) FieldType(i int) Type   { panic("not implemented") }
func (t *TypeImpl) FieldOff(i int) int64   { panic("not implemented") }
func (t *TypeImpl) FieldName(i int) string { panic("not implemented") }
func (t *TypeImpl) NumElem() int64         { panic("not implemented") }

func (t *TypeImpl) Equal(u Type) bool {
	x, ok := u.(*TypeImpl)
	if !ok {
		return false
	}
	return x == t
}

func (t *TypeImpl) Compare(u Type) Cmp {
	x, ok := u.(*TypeImpl)
	// ssa.CompilerType < ssa.TypeImpl < gc.Type
	if !ok {
		_, ok := u.(*CompilerType)
		if ok {
			return CMPgt
		}
		return CMPlt
	}
	if t == x {
		return CMPeq
	}
	if t.Name < x.Name {
		return CMPlt
	}
	if t.Name > x.Name {
		return CMPgt
	}
	return CMPeq

}

var (
	// shortcuts for commonly used basic types
	TypeInt8       = &TypeImpl{Size_: 1, Align: 1, Integer: true, Signed: true, Name: "int8"}
	TypeInt16      = &TypeImpl{Size_: 2, Align: 2, Integer: true, Signed: true, Name: "int16"}
	TypeInt32      = &TypeImpl{Size_: 4, Align: 4, Integer: true, Signed: true, Name: "int32"}
	TypeInt64      = &TypeImpl{Size_: 8, Align: 8, Integer: true, Signed: true, Name: "int64"}
	TypeFloat32    = &TypeImpl{Size_: 4, Align: 4, Float: true, Name: "float32"}
	TypeFloat64    = &TypeImpl{Size_: 8, Align: 8, Float: true, Name: "float64"}
	TypeComplex64  = &TypeImpl{Size_: 8, Align: 4, Complex: true, Name: "complex64"}
	TypeComplex128 = &TypeImpl{Size_: 16, Align: 8, Complex: true, Name: "complex128"}
	TypeUInt8      = &TypeImpl{Size_: 1, Align: 1, Integer: true, Name: "uint8"}
	TypeUInt16     = &TypeImpl{Size_: 2, Align: 2, Integer: true, Name: "uint16"}
	TypeUInt32     = &TypeImpl{Size_: 4, Align: 4, Integer: true, Name: "uint32"}
	TypeUInt64     = &TypeImpl{Size_: 8, Align: 8, Integer: true, Name: "uint64"}
	TypeBool       = &TypeImpl{Size_: 1, Align: 1, Boolean: true, Name: "bool"}
	TypeBytePtr    = &TypeImpl{Size_: 8, Align: 8, Ptr: true, Name: "*byte"}
	TypeInt64Ptr   = &TypeImpl{Size_: 8, Align: 8, Ptr: true, Name: "*int64"}
)
