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
	IsPtr() bool
	IsString() bool
	IsSlice() bool
	IsArray() bool
	IsStruct() bool
	IsInterface() bool

	IsMemory() bool // special ssa-package-only types
	IsFlags() bool
	IsVoid() bool

	Elem() Type  // given []T or *T or [n]T, return T
	PtrTo() Type // given T, return *T

	NumFields() int64       // # of fields of a struct
	FieldType(i int64) Type // type of ith field of the struct
	FieldOff(i int64) int64 // offset of ith field of the struct

	NumElem() int64 // # of elements of an array

	String() string
	SimpleString() string // a coarser generic description of T, e.g. T's underlying type
	Equal(Type) bool
}

// Special compiler-only types.
type CompilerType struct {
	Name   string
	Memory bool
	Flags  bool
	Void   bool
}

func (t *CompilerType) Size() int64            { return 0 } // Size in bytes
func (t *CompilerType) Alignment() int64       { return 0 }
func (t *CompilerType) IsBoolean() bool        { return false }
func (t *CompilerType) IsInteger() bool        { return false }
func (t *CompilerType) IsSigned() bool         { return false }
func (t *CompilerType) IsFloat() bool          { return false }
func (t *CompilerType) IsComplex() bool        { return false }
func (t *CompilerType) IsPtr() bool            { return false }
func (t *CompilerType) IsString() bool         { return false }
func (t *CompilerType) IsSlice() bool          { return false }
func (t *CompilerType) IsArray() bool          { return false }
func (t *CompilerType) IsStruct() bool         { return false }
func (t *CompilerType) IsInterface() bool      { return false }
func (t *CompilerType) IsMemory() bool         { return t.Memory }
func (t *CompilerType) IsFlags() bool          { return t.Flags }
func (t *CompilerType) IsVoid() bool           { return t.Void }
func (t *CompilerType) String() string         { return t.Name }
func (t *CompilerType) SimpleString() string   { return t.Name }
func (t *CompilerType) Elem() Type             { panic("not implemented") }
func (t *CompilerType) PtrTo() Type            { panic("not implemented") }
func (t *CompilerType) NumFields() int64       { panic("not implemented") }
func (t *CompilerType) FieldType(i int64) Type { panic("not implemented") }
func (t *CompilerType) FieldOff(i int64) int64 { panic("not implemented") }
func (t *CompilerType) NumElem() int64         { panic("not implemented") }

func (t *CompilerType) Equal(u Type) bool {
	x, ok := u.(*CompilerType)
	if !ok {
		return false
	}
	return x == t
}

var (
	TypeInvalid = &CompilerType{Name: "invalid"}
	TypeMem     = &CompilerType{Name: "mem", Memory: true}
	TypeFlags   = &CompilerType{Name: "flags", Flags: true}
	TypeVoid    = &CompilerType{Name: "void", Void: true}
)
