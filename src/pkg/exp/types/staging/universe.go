// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements the universe and unsafe package scopes.

package types

import (
	"go/ast"
	"strings"
)

var (
	aType            implementsType
	Universe, unsafe *ast.Scope
	Unsafe           *ast.Object // package unsafe
)

// Predeclared types, indexed by BasicKind.
var Typ = [...]*Basic{
	Invalid: {aType, Invalid, 0, "invalid type"},

	Bool:          {aType, Bool, IsBoolean, "bool"},
	Int:           {aType, Int, IsInteger, "int"},
	Int8:          {aType, Int8, IsInteger, "int8"},
	Int16:         {aType, Int16, IsInteger, "int16"},
	Int32:         {aType, Int32, IsInteger, "int32"},
	Int64:         {aType, Int64, IsInteger, "int64"},
	Uint:          {aType, Uint, IsInteger | IsUnsigned, "uint"},
	Uint8:         {aType, Uint8, IsInteger | IsUnsigned, "uint8"},
	Uint16:        {aType, Uint16, IsInteger | IsUnsigned, "uint16"},
	Uint32:        {aType, Uint32, IsInteger | IsUnsigned, "uint32"},
	Uint64:        {aType, Uint64, IsInteger | IsUnsigned, "uint64"},
	Uintptr:       {aType, Uintptr, IsInteger | IsUnsigned, "uintptr"},
	Float32:       {aType, Float32, IsFloat, "float32"},
	Float64:       {aType, Float64, IsFloat, "float64"},
	Complex64:     {aType, Complex64, IsComplex, "complex64"},
	Complex128:    {aType, Complex128, IsComplex, "complex128"},
	String:        {aType, String, IsString, "string"},
	UnsafePointer: {aType, UnsafePointer, 0, "Pointer"},

	UntypedBool:    {aType, UntypedBool, IsBoolean | IsUntyped, "untyped boolean"},
	UntypedInt:     {aType, UntypedInt, IsInteger | IsUntyped, "untyped integer"},
	UntypedRune:    {aType, UntypedRune, IsInteger | IsUntyped, "untyped rune"},
	UntypedFloat:   {aType, UntypedFloat, IsFloat | IsUntyped, "untyped float"},
	UntypedComplex: {aType, UntypedComplex, IsComplex | IsUntyped, "untyped complex"},
	UntypedString:  {aType, UntypedString, IsString | IsUntyped, "untyped string"},
	UntypedNil:     {aType, UntypedNil, IsUntyped, "untyped nil"},
}

var aliases = [...]*Basic{
	{aType, Uint8, IsInteger | IsUnsigned, "byte"},
	{aType, Rune, IsInteger, "rune"},
}

var predeclaredConstants = [...]*struct {
	kind BasicKind
	name string
	val  interface{}
}{
	{UntypedBool, "true", true},
	{UntypedBool, "false", false},
	{UntypedInt, "iota", int64(0)},
	{UntypedNil, "nil", nil},
}

var predeclaredFunctions = [...]*builtin{
	{aType, _Append, "append", 1, true},
	{aType, _Cap, "cap", 1, false},
	{aType, _Close, "close", 1, false},
	{aType, _Complex, "complex", 2, false},
	{aType, _Copy, "copy", 2, false},
	{aType, _Delete, "delete", 2, false},
	{aType, _Imag, "imag", 1, false},
	{aType, _Len, "len", 1, false},
	{aType, _Make, "make", 1, true},
	{aType, _New, "new", 1, false},
	{aType, _Panic, "panic", 1, false},
	{aType, _Print, "print", 1, true},
	{aType, _Println, "println", 1, true},
	{aType, _Real, "real", 1, false},
	{aType, _Recover, "recover", 0, false},

	{aType, _Alignof, "Alignof", 1, false},
	{aType, _Offsetof, "Offsetof", 1, false},
	{aType, _Sizeof, "Sizeof", 1, false},
}

// commonly used types
var (
	emptyInterface = new(Interface)
)

// commonly used constants
var (
	universeIota *ast.Object
)

func init() {
	// Universe scope
	Universe = ast.NewScope(nil)

	// unsafe package and its scope
	unsafe = ast.NewScope(nil)
	Unsafe = ast.NewObj(ast.Pkg, "unsafe")
	Unsafe.Data = unsafe

	// predeclared types
	for _, t := range Typ {
		def(ast.Typ, t.Name).Type = t
	}
	for _, t := range aliases {
		def(ast.Typ, t.Name).Type = t
	}

	// error type
	{
		obj := def(ast.Typ, "error")
		// TODO(gri) set up correct interface type
		typ := &NamedType{Underlying: &Interface{}, Obj: obj}
		obj.Type = typ
	}

	// predeclared constants
	for _, t := range predeclaredConstants {
		obj := def(ast.Con, t.name)
		obj.Type = Typ[t.kind]
		obj.Data = t.val
	}

	// predeclared functions
	for _, f := range predeclaredFunctions {
		def(ast.Fun, f.name).Type = f
	}

	universeIota = Universe.Lookup("iota")
}

// Objects with names containing blanks are internal and not entered into
// a scope. Objects with exported names are inserted in the unsafe package
// scope; other objects are inserted in the universe scope.
//
func def(kind ast.ObjKind, name string) *ast.Object {
	obj := ast.NewObj(kind, name)
	// insert non-internal objects into respective scope
	if strings.Index(name, " ") < 0 {
		scope := Universe
		// exported identifiers go into package unsafe
		if ast.IsExported(name) {
			scope = unsafe
		}
		if scope.Insert(obj) != nil {
			panic("internal error: double declaration")
		}
		obj.Decl = scope
	}
	return obj
}
