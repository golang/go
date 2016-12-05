// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// TODO: use go/types instead?

// A type interface used to import cmd/internal/gc:Type
// Type instances are not guaranteed to be canonical.
type Type interface {
	Size() int64 // return the size in bytes
	Alignment() int64

	IsBoolean() bool // is a named or unnamed boolean type
	IsInteger() bool //  ... ditto for the others
	IsSigned() bool
	IsFloat() bool
	IsComplex() bool
	IsPtrShaped() bool
	IsString() bool
	IsSlice() bool
	IsArray() bool
	IsStruct() bool
	IsInterface() bool

	IsMemory() bool // special ssa-package-only types
	IsFlags() bool
	IsVoid() bool
	IsTuple() bool

	ElemType() Type // given []T or *T or [n]T, return T
	PtrTo() Type    // given T, return *T

	NumFields() int         // # of fields of a struct
	FieldType(i int) Type   // type of ith field of the struct or ith part of a tuple
	FieldOff(i int) int64   // offset of ith field of the struct
	FieldName(i int) string // name of ith field of the struct

	NumElem() int64 // # of elements of an array

	String() string
	SimpleString() string // a coarser generic description of T, e.g. T's underlying type
	Compare(Type) Cmp     // compare types, returning one of CMPlt, CMPeq, CMPgt.
}

// Special compiler-only types.
type CompilerType struct {
	Name   string
	size   int64
	Memory bool
	Flags  bool
	Void   bool
	Int128 bool
}

func (t *CompilerType) Size() int64            { return t.size } // Size in bytes
func (t *CompilerType) Alignment() int64       { return 0 }
func (t *CompilerType) IsBoolean() bool        { return false }
func (t *CompilerType) IsInteger() bool        { return false }
func (t *CompilerType) IsSigned() bool         { return false }
func (t *CompilerType) IsFloat() bool          { return false }
func (t *CompilerType) IsComplex() bool        { return false }
func (t *CompilerType) IsPtrShaped() bool      { return false }
func (t *CompilerType) IsString() bool         { return false }
func (t *CompilerType) IsSlice() bool          { return false }
func (t *CompilerType) IsArray() bool          { return false }
func (t *CompilerType) IsStruct() bool         { return false }
func (t *CompilerType) IsInterface() bool      { return false }
func (t *CompilerType) IsMemory() bool         { return t.Memory }
func (t *CompilerType) IsFlags() bool          { return t.Flags }
func (t *CompilerType) IsVoid() bool           { return t.Void }
func (t *CompilerType) IsTuple() bool          { return false }
func (t *CompilerType) String() string         { return t.Name }
func (t *CompilerType) SimpleString() string   { return t.Name }
func (t *CompilerType) ElemType() Type         { panic("not implemented") }
func (t *CompilerType) PtrTo() Type            { panic("not implemented") }
func (t *CompilerType) NumFields() int         { panic("not implemented") }
func (t *CompilerType) FieldType(i int) Type   { panic("not implemented") }
func (t *CompilerType) FieldOff(i int) int64   { panic("not implemented") }
func (t *CompilerType) FieldName(i int) string { panic("not implemented") }
func (t *CompilerType) NumElem() int64         { panic("not implemented") }

type TupleType struct {
	first  Type
	second Type
	// Any tuple with a memory type must put that memory type second.
}

func (t *TupleType) Size() int64          { panic("not implemented") }
func (t *TupleType) Alignment() int64     { panic("not implemented") }
func (t *TupleType) IsBoolean() bool      { return false }
func (t *TupleType) IsInteger() bool      { return false }
func (t *TupleType) IsSigned() bool       { return false }
func (t *TupleType) IsFloat() bool        { return false }
func (t *TupleType) IsComplex() bool      { return false }
func (t *TupleType) IsPtrShaped() bool    { return false }
func (t *TupleType) IsString() bool       { return false }
func (t *TupleType) IsSlice() bool        { return false }
func (t *TupleType) IsArray() bool        { return false }
func (t *TupleType) IsStruct() bool       { return false }
func (t *TupleType) IsInterface() bool    { return false }
func (t *TupleType) IsMemory() bool       { return false }
func (t *TupleType) IsFlags() bool        { return false }
func (t *TupleType) IsVoid() bool         { return false }
func (t *TupleType) IsTuple() bool        { return true }
func (t *TupleType) String() string       { return t.first.String() + "," + t.second.String() }
func (t *TupleType) SimpleString() string { return "Tuple" }
func (t *TupleType) ElemType() Type       { panic("not implemented") }
func (t *TupleType) PtrTo() Type          { panic("not implemented") }
func (t *TupleType) NumFields() int       { panic("not implemented") }
func (t *TupleType) FieldType(i int) Type {
	switch i {
	case 0:
		return t.first
	case 1:
		return t.second
	default:
		panic("bad tuple index")
	}
}
func (t *TupleType) FieldOff(i int) int64   { panic("not implemented") }
func (t *TupleType) FieldName(i int) string { panic("not implemented") }
func (t *TupleType) NumElem() int64         { panic("not implemented") }

// Cmp is a comparison between values a and b.
// -1 if a < b
//  0 if a == b
//  1 if a > b
type Cmp int8

const (
	CMPlt = Cmp(-1)
	CMPeq = Cmp(0)
	CMPgt = Cmp(1)
)

func (t *CompilerType) Compare(u Type) Cmp {
	x, ok := u.(*CompilerType)
	// ssa.CompilerType is smaller than any other type
	if !ok {
		return CMPlt
	}
	if t == x {
		return CMPeq
	}
	// desire fast sorting, not pretty sorting.
	if len(t.Name) == len(x.Name) {
		if t.Name == x.Name {
			return CMPeq
		}
		if t.Name < x.Name {
			return CMPlt
		}
		return CMPgt
	}
	if len(t.Name) > len(x.Name) {
		return CMPgt
	}
	return CMPlt
}

func (t *TupleType) Compare(u Type) Cmp {
	// ssa.TupleType is greater than ssa.CompilerType
	if _, ok := u.(*CompilerType); ok {
		return CMPgt
	}
	// ssa.TupleType is smaller than any other type
	x, ok := u.(*TupleType)
	if !ok {
		return CMPlt
	}
	if t == x {
		return CMPeq
	}
	if c := t.first.Compare(x.first); c != CMPeq {
		return c
	}
	return t.second.Compare(x.second)
}

var (
	TypeInvalid = &CompilerType{Name: "invalid"}
	TypeMem     = &CompilerType{Name: "mem", Memory: true}
	TypeFlags   = &CompilerType{Name: "flags", Flags: true}
	TypeVoid    = &CompilerType{Name: "void", Void: true}
	TypeInt128  = &CompilerType{Name: "int128", size: 16, Int128: true}
)

func MakeTuple(t0, t1 Type) *TupleType {
	return &TupleType{first: t0, second: t1}
}
