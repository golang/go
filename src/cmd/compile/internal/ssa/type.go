// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// TODO: use go/types instead?

// A type interface used to import cmd/internal/gc:Type
// Type instances are not guaranteed to be canonical.
type Type interface {
	Size() int64 // return the size in bytes

	IsBoolean() bool // is a named or unnamed boolean type
	IsInteger() bool //  ... ditto for the others
	IsSigned() bool
	IsFloat() bool
	IsPtr() bool

	IsMemory() bool // special ssa-package-only types
	IsFlags() bool

	Elem() Type  // given []T or *T, return T
	PtrTo() Type // given T, return *T

	String() string
}

// Stub implementation for now, until we are completely using ../gc:Type
type TypeImpl struct {
	Size_   int64
	Boolean bool
	Integer bool
	Signed  bool
	Float   bool
	Ptr     bool

	Memory bool
	Flags  bool

	Name string
}

func (t *TypeImpl) Size() int64     { return t.Size_ }
func (t *TypeImpl) IsBoolean() bool { return t.Boolean }
func (t *TypeImpl) IsInteger() bool { return t.Integer }
func (t *TypeImpl) IsSigned() bool  { return t.Signed }
func (t *TypeImpl) IsFloat() bool   { return t.Float }
func (t *TypeImpl) IsPtr() bool     { return t.Ptr }
func (t *TypeImpl) IsMemory() bool  { return t.Memory }
func (t *TypeImpl) IsFlags() bool   { return t.Flags }
func (t *TypeImpl) String() string  { return t.Name }
func (t *TypeImpl) Elem() Type      { panic("not implemented"); return nil }
func (t *TypeImpl) PtrTo() Type     { panic("not implemented"); return nil }

var (
	// shortcuts for commonly used basic types
	TypeInt8   = &TypeImpl{Size_: 1, Integer: true, Signed: true, Name: "int8"}
	TypeInt16  = &TypeImpl{Size_: 2, Integer: true, Signed: true, Name: "int16"}
	TypeInt32  = &TypeImpl{Size_: 4, Integer: true, Signed: true, Name: "int32"}
	TypeInt64  = &TypeImpl{Size_: 8, Integer: true, Signed: true, Name: "int64"}
	TypeUInt8  = &TypeImpl{Size_: 1, Integer: true, Name: "uint8"}
	TypeUInt16 = &TypeImpl{Size_: 2, Integer: true, Name: "uint16"}
	TypeUInt32 = &TypeImpl{Size_: 4, Integer: true, Name: "uint32"}
	TypeUInt64 = &TypeImpl{Size_: 8, Integer: true, Name: "uint64"}
	TypeBool   = &TypeImpl{Size_: 1, Boolean: true, Name: "bool"}
	//TypeString = types.Typ[types.String]

	TypeInvalid = &TypeImpl{Name: "invalid"}

	// Additional compiler-only types go here.
	TypeMem   = &TypeImpl{Memory: true, Name: "mem"}
	TypeFlags = &TypeImpl{Flags: true, Name: "flags"}
)
