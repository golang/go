// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package types declares the data structures for representing
// Go types and implements typechecking of an *ast.Package.
//
// PACKAGE UNDER CONSTRUCTION. ANY AND ALL PARTS MAY CHANGE.
//
package types

import (
	"go/ast"
	"go/token"
	"sort"
)

// Check typechecks the given package pkg and augments the AST by
// assigning types to all ast.Objects. Check can be used in two
// different modes:
//
// 1) If a nil types map is provided, Check typechecks the entire
// package. If no error is returned, the package source code has
// no type errors. 
//
// 2) If a non-nil types map is provided, Check operates like in
// mode 1) but also records the types for all expressions in the
// map. Pre-existing expression types in the map are replaced if
// the expression appears in the AST.
//
func Check(fset *token.FileSet, pkg *ast.Package, types map[ast.Expr]Type) error {
	return check(fset, pkg, types)
}

// All types implement the Type interface.
// TODO(gri) Eventually determine what common Type functionality should be exported.
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

	IsOrdered = IsInteger | IsFloat | IsString
	IsNumeric = IsInteger | IsFloat | IsComplex
)

// A Basic represents a basic type.
type Basic struct {
	implementsType
	Kind BasicKind
	Info BasicInfo
	Size int64 // > 0 if valid
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

type StructField struct {
	Name        string // unqualified type name for anonymous fields
	Type        Type
	Tag         string
	IsAnonymous bool
}

// A Struct represents a struct type struct{...}.
type Struct struct {
	implementsType
	Fields []*StructField
}

// A Pointer represents a pointer type *Base.
type Pointer struct {
	implementsType
	Base Type
}

// A tuple represents a multi-value function return.
// TODO(gri) use better name to avoid confusion (Go doesn't have tuples).
type tuple struct {
	implementsType
	list []Type
}

// A Signature represents a user-defined function type func(...) (...).
// TODO(gri) consider using "tuples" to represent parameters and results (see comment on tuples).
type Signature struct {
	implementsType
	Recv       *ast.Object // nil if not a method
	Params     ObjList     // (incoming) parameters from left to right; or nil
	Results    ObjList     // (outgoing) results from left to right; or nil
	IsVariadic bool        // true if the last parameter's type is of the form ...T
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

// An Interface represents an interface type interface{...}.
type Interface struct {
	implementsType
	Methods ObjList // interface methods sorted by name; or nil
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
	Obj        *ast.Object // corresponding declared object
	Underlying Type        // nil if not fully declared yet, never a *NamedType
	Methods    ObjList     // associated methods; or nil
}

// An ObjList represents an ordered (in some fashion) list of objects.
type ObjList []*ast.Object

// ObjList implements sort.Interface.
func (list ObjList) Len() int           { return len(list) }
func (list ObjList) Less(i, j int) bool { return list[i].Name < list[j].Name }
func (list ObjList) Swap(i, j int)      { list[i], list[j] = list[j], list[i] }

// Sort sorts an object list by object name.
func (list ObjList) Sort() { sort.Sort(list) }

// All concrete types embed implementsType which
// ensures that all types implement the Type interface.
type implementsType struct{}

func (*implementsType) aType() {}
