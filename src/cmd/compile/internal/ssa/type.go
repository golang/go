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
	IsPtr() bool
	IsString() bool

	IsMemory() bool // special ssa-package-only types
	IsFlags() bool

	Elem() Type  // given []T or *T, return T
	PtrTo() Type // given T, return *T

	String() string
	Equal(Type) bool
}

// Special compiler-only types.
type CompilerType struct {
	Name   string
	Memory bool
	Flags  bool
}

func (t *CompilerType) Size() int64      { return 0 }
func (t *CompilerType) Alignment() int64 { return 0 }
func (t *CompilerType) IsBoolean() bool  { return false }
func (t *CompilerType) IsInteger() bool  { return false }
func (t *CompilerType) IsSigned() bool   { return false }
func (t *CompilerType) IsFloat() bool    { return false }
func (t *CompilerType) IsPtr() bool      { return false }
func (t *CompilerType) IsString() bool   { return false }
func (t *CompilerType) IsMemory() bool   { return t.Memory }
func (t *CompilerType) IsFlags() bool    { return t.Flags }
func (t *CompilerType) String() string   { return t.Name }
func (t *CompilerType) Elem() Type       { panic("not implemented") }
func (t *CompilerType) PtrTo() Type      { panic("not implemented") }

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
)
