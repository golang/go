// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import "go/ast"

// All types implement the Type interface.
type Type interface {
	aType()
}

// BasicKind describes the kind of basic type.
type BasicKind int

const (
	Invalid BasicKind = iota // type is invalid

	// predeclared types
	Bool
	Int
	Int8
	Int16
	Int32
	Int64
	Uint
	Uint8
	Uint16
	Uint32
	Uint64
	Uintptr
	Float32
	Float64
	Complex64
	Complex128
	String
	UnsafePointer

	// types for untyped values
	UntypedBool
	UntypedInt
	UntypedRune
	UntypedFloat
	UntypedComplex
	UntypedString
	UntypedNil

	// aliases
	Byte = Uint8
	Rune = Int32
)

// BasicInfo is a set of flags describing properties of a basic type.
type BasicInfo int

// Properties of basic types.
const (
	IsBoolean BasicInfo = 1 << iota
	IsInteger
	IsUnsigned
	IsFloat
	IsComplex
	IsString
	IsUntyped

	IsOrdered   = IsInteger | IsFloat | IsString
	IsNumeric   = IsInteger | IsFloat | IsComplex
	IsConstType = IsBoolean | IsNumeric | IsString
)

// A Basic represents a basic type.
type Basic struct {
	implementsType
	Kind BasicKind
	Info BasicInfo
	Size int64
	Name string
}

// An Array represents an array type [Len]Elt.
type Array struct {
	implementsType
	Len int64
	Elt Type
}

// A Slice represents a slice type []Elt.
type Slice struct {
	implementsType
	Elt Type
}

// A Field represents a field of a struct.
type Field struct {
	Name        string // unqualified type name for anonymous fields
	Type        Type
	Tag         string
	IsAnonymous bool
}

// A Struct represents a struct type struct{...}.
type Struct struct {
	implementsType
	Fields []*Field
}

func (typ *Struct) fieldIndex(name string) int {
	for i, f := range typ.Fields {
		if f.Name == name {
			return i
		}
	}
	return -1
}

// A Pointer represents a pointer type *Base.
type Pointer struct {
	implementsType
	Base Type
}

// A Variable represents a variable (including function parameters and results).
type Var struct {
	Name string
	Type Type
}

// A Result represents a (multi-value) function call result.
type Result struct {
	implementsType
	Values []*Var // Signature.Results of the function called
}

// A Signature represents a user-defined function type func(...) (...).
type Signature struct {
	implementsType
	Recv       *Var   // nil if not a method
	Params     []*Var // (incoming) parameters from left to right; or nil
	Results    []*Var // (outgoing) results from left to right; or nil
	IsVariadic bool   // true if the last parameter's type is of the form ...T
}

// builtinId is an id of a builtin function.
type builtinId int

// Predeclared builtin functions.
const (
	// Universe scope
	_Append builtinId = iota
	_Cap
	_Close
	_Complex
	_Copy
	_Delete
	_Imag
	_Len
	_Make
	_New
	_Panic
	_Print
	_Println
	_Real
	_Recover

	// Unsafe package
	_Alignof
	_Offsetof
	_Sizeof

	// Testing support
	_Assert
	_Trace
)

// A builtin represents the type of a built-in function.
type builtin struct {
	implementsType
	id          builtinId
	name        string
	nargs       int // number of arguments (minimum if variadic)
	isVariadic  bool
	isStatement bool // true if the built-in is valid as an expression statement
}

// A Method represents a method of an interface.
type Method struct {
	Name string
	Type *Signature
}

// An Interface represents an interface type interface{...}.
type Interface struct {
	implementsType
	Methods []*Method // TODO(gri) consider keeping them in sorted order
}

// A Map represents a map type map[Key]Elt.
type Map struct {
	implementsType
	Key, Elt Type
}

// A Chan represents a channel type chan Elt, <-chan Elt, or chan<-Elt.
type Chan struct {
	implementsType
	Dir ast.ChanDir
	Elt Type
}

// A NamedType represents a named type as declared in a type declaration.
type NamedType struct {
	implementsType
	Obj        *ast.Object // corresponding declared object; Obj.Data.(*ast.Scope) contains methods, if any
	Underlying Type        // nil if not fully declared yet; never a *NamedType
}

// All concrete types embed implementsType which
// ensures that all types implement the Type interface.
type implementsType struct{}

func (*implementsType) aType() {}
