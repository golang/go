// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import "go/ast"

// All types implement the Type interface.
type Type interface {
	String() string
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
	Kind BasicKind
	Info BasicInfo
	size int64 // use DefaultSizeof to get size
	Name string
}

// An Array represents an array type [Len]Elt.
type Array struct {
	Len int64
	Elt Type
}

// A Slice represents a slice type []Elt.
type Slice struct {
	Elt Type
}

// A QualifiedName is a name qualified with the package that declared the name.
// Note: Pkg may be a fake package (no name, no scope) because the GC compiler's
//       export information doesn't provide full information in some cases.
// TODO(gri): Should change Pkg to PkgPath since it's the only thing we care about.
type QualifiedName struct {
	Pkg  *Package // nil only for predeclared error.Error (exported)
	Name string   // unqualified type name for anonymous fields
}

// IsSame reports whether p and q are the same.
func (p QualifiedName) IsSame(q QualifiedName) bool {
	// spec:
	// "Two identifiers are different if they are spelled differently,
	// or if they appear in different packages and are not exported.
	// Otherwise, they are the same."
	if p.Name != q.Name {
		return false
	}
	// p.Name == q.Name
	return ast.IsExported(p.Name) || p.Pkg.Path == q.Pkg.Path
}

// A Field represents a field of a struct.
type Field struct {
	QualifiedName
	Type        Type
	Tag         string
	IsAnonymous bool
}

// A Struct represents a struct type struct{...}.
type Struct struct {
	Fields  []*Field
	offsets []int64 // field offsets in bytes, lazily computed
}

func (typ *Struct) fieldIndex(name QualifiedName) int {
	for i, f := range typ.Fields {
		if f.QualifiedName.IsSame(name) {
			return i
		}
	}
	return -1
}

// A Pointer represents a pointer type *Base.
type Pointer struct {
	Base Type
}

// A Result represents a (multi-value) function call result.
type Result struct {
	Values []*Var // Signature.Results of the function called
}

// A Signature represents a user-defined function type func(...) (...).
type Signature struct {
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
	id          builtinId
	name        string
	nargs       int // number of arguments (minimum if variadic)
	isVariadic  bool
	isStatement bool // true if the built-in is valid as an expression statement
}

// A Method represents a method.
type Method struct {
	QualifiedName
	Type *Signature
}

// An Interface represents an interface type interface{...}.
type Interface struct {
	Methods []*Method // TODO(gri) consider keeping them in sorted order
}

// A Map represents a map type map[Key]Elt.
type Map struct {
	Key, Elt Type
}

// A Chan represents a channel type chan Elt, <-chan Elt, or chan<-Elt.
type Chan struct {
	Dir ast.ChanDir
	Elt Type
}

// A NamedType represents a named type as declared in a type declaration.
type NamedType struct {
	Obj        *TypeName // corresponding declared object
	Underlying Type      // nil if not fully declared yet; never a *NamedType
	Methods    []*Method // TODO(gri) consider keeping them in sorted order
}

func (*Basic) aType()     {}
func (*Array) aType()     {}
func (*Slice) aType()     {}
func (*Struct) aType()    {}
func (*Pointer) aType()   {}
func (*Result) aType()    {}
func (*Signature) aType() {}
func (*builtin) aType()   {}
func (*Interface) aType() {}
func (*Map) aType()       {}
func (*Chan) aType()      {}
func (*NamedType) aType() {}
