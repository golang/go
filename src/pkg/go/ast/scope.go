// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements scopes, the objects they contain,
// and object types.

package ast

// A Scope maintains the set of named language entities declared
// in the scope and a link to the immediately surrounding (outer)
// scope.
//
type Scope struct {
	Outer   *Scope
	Objects []*Object // in declaration order
	// Implementation note: In some cases (struct fields,
	// function parameters) we need the source order of
	// variables. Thus for now, we store scope entries
	// in a linear list. If scopes become very large
	// (say, for packages), we may need to change this
	// to avoid slow lookups.
}


// NewScope creates a new scope nested in the outer scope.
func NewScope(outer *Scope) *Scope {
	const n = 4 // initial scope capacity, must be > 0
	return &Scope{outer, make([]*Object, 0, n)}
}


// Lookup returns the object with the given name if it is
// found in scope s, otherwise it returns nil. Outer scopes
// are ignored.
//
// Lookup always returns nil if name is "_", even if the scope
// contains objects with that name.
//
func (s *Scope) Lookup(name string) *Object {
	if name != "_" {
		for _, obj := range s.Objects {
			if obj.Name == name {
				return obj
			}
		}
	}
	return nil
}


// Insert attempts to insert a named object into the scope s.
// If the scope does not contain an object with that name yet
// or if the object is named "_", Insert inserts the object
// and returns it. Otherwise, Insert leaves the scope unchanged
// and returns the object found in the scope instead.
//
func (s *Scope) Insert(obj *Object) *Object {
	alt := s.Lookup(obj.Name)
	if alt == nil {
		s.append(obj)
		alt = obj
	}
	return alt
}


func (s *Scope) append(obj *Object) {
	s.Objects = append(s.Objects, obj)
}

// ----------------------------------------------------------------------------
// Objects

// An Object describes a language entity such as a package,
// constant, type, variable, or function (incl. methods).
//
type Object struct {
	Kind Kind
	Name string // declared name
	Type *Type
	Decl interface{} // corresponding Field, XxxSpec or FuncDecl
	N    int         // value of iota for this declaration
}


// NewObj creates a new object of a given kind and name.
func NewObj(kind Kind, name string) *Object {
	return &Object{Kind: kind, Name: name}
}


// Kind describes what an object represents.
type Kind int

// The list of possible Object kinds.
const (
	Bad Kind = iota // for error handling
	Pkg             // package
	Con             // constant
	Typ             // type
	Var             // variable
	Fun             // function or method
)


var objKindStrings = [...]string{
	Bad: "bad",
	Pkg: "package",
	Con: "const",
	Typ: "type",
	Var: "var",
	Fun: "func",
}


func (kind Kind) String() string { return objKindStrings[kind] }


// IsExported returns whether obj is exported.
func (obj *Object) IsExported() bool { return IsExported(obj.Name) }


// ----------------------------------------------------------------------------
// Types

// A Type represents a Go type.
type Type struct {
	Form     Form
	Obj      *Object // corresponding type name, or nil
	Scope    *Scope  // fields and methods, always present
	N        uint    // basic type id, array length, number of function results, or channel direction
	Key, Elt *Type   // map key and array, pointer, slice, map or channel element
	Params   *Scope  // function (receiver, input and result) parameters, tuple expressions (results of function calls), or nil
	Expr     Expr    // corresponding AST expression
}


// NewType creates a new type of a given form.
func NewType(form Form) *Type {
	return &Type{Form: form, Scope: NewScope(nil)}
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
