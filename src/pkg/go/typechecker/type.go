// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typechecker

import "go/ast"

// A Type represents a Go type.
type Type struct {
	Form     Form
	Obj      *ast.Object // corresponding type name, or nil
	Scope    *ast.Scope  // fields and methods, always present
	N        uint        // basic type id, array length, number of function results, or channel direction
	Key, Elt *Type       // map key and array, pointer, slice, map or channel element
	Params   *ast.Scope  // function (receiver, input and result) parameters, tuple expressions (results of function calls), or nil
	Expr     ast.Expr    // corresponding AST expression
}

// NewType creates a new type of a given form.
func NewType(form Form) *Type {
	return &Type{Form: form, Scope: ast.NewScope(nil)}
}

// Form describes the form of a type.
type Form int

// The list of possible type forms.
const (
	BadType    Form = iota // for error handling
	Unresolved             // type not fully setup
	Basic
	Array
	Struct
	Pointer
	Function
	Method
	Interface
	Slice
	Map
	Channel
	Tuple
)

var formStrings = [...]string{
	BadType:    "badType",
	Unresolved: "unresolved",
	Basic:      "basic",
	Array:      "array",
	Struct:     "struct",
	Pointer:    "pointer",
	Function:   "function",
	Method:     "method",
	Interface:  "interface",
	Slice:      "slice",
	Map:        "map",
	Channel:    "channel",
	Tuple:      "tuple",
}

func (form Form) String() string { return formStrings[form] }

// The list of basic type id's.
const (
	Bool = iota
	Byte
	Uint
	Int
	Float
	Complex
	Uintptr
	String

	Uint8
	Uint16
	Uint32
	Uint64

	Int8
	Int16
	Int32
	Int64

	Float32
	Float64

	Complex64
	Complex128

	// TODO(gri) ideal types are missing
)

var BasicTypes = map[uint]string{
	Bool:    "bool",
	Byte:    "byte",
	Uint:    "uint",
	Int:     "int",
	Float:   "float",
	Complex: "complex",
	Uintptr: "uintptr",
	String:  "string",

	Uint8:  "uint8",
	Uint16: "uint16",
	Uint32: "uint32",
	Uint64: "uint64",

	Int8:  "int8",
	Int16: "int16",
	Int32: "int32",
	Int64: "int64",

	Float32: "float32",
	Float64: "float64",

	Complex64:  "complex64",
	Complex128: "complex128",
}
