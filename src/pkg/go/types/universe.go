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
	Universe     *Scope
	Unsafe       *Package
	universeIota *Const
)

// Predeclared types, indexed by BasicKind.
var Typ = [...]*Basic{
	Invalid: {Invalid, 0, 0, "invalid type"},

	Bool:          {Bool, IsBoolean, 1, "bool"},
	Int:           {Int, IsInteger, 0, "int"},
	Int8:          {Int8, IsInteger, 1, "int8"},
	Int16:         {Int16, IsInteger, 2, "int16"},
	Int32:         {Int32, IsInteger, 4, "int32"},
	Int64:         {Int64, IsInteger, 8, "int64"},
	Uint:          {Uint, IsInteger | IsUnsigned, 0, "uint"},
	Uint8:         {Uint8, IsInteger | IsUnsigned, 1, "uint8"},
	Uint16:        {Uint16, IsInteger | IsUnsigned, 2, "uint16"},
	Uint32:        {Uint32, IsInteger | IsUnsigned, 4, "uint32"},
	Uint64:        {Uint64, IsInteger | IsUnsigned, 8, "uint64"},
	Uintptr:       {Uintptr, IsInteger | IsUnsigned, 0, "uintptr"},
	Float32:       {Float32, IsFloat, 4, "float32"},
	Float64:       {Float64, IsFloat, 8, "float64"},
	Complex64:     {Complex64, IsComplex, 8, "complex64"},
	Complex128:    {Complex128, IsComplex, 16, "complex128"},
	String:        {String, IsString, 0, "string"},
	UnsafePointer: {UnsafePointer, 0, 0, "Pointer"},

	UntypedBool:    {UntypedBool, IsBoolean | IsUntyped, 0, "untyped boolean"},
	UntypedInt:     {UntypedInt, IsInteger | IsUntyped, 0, "untyped integer"},
	UntypedRune:    {UntypedRune, IsInteger | IsUntyped, 0, "untyped rune"},
	UntypedFloat:   {UntypedFloat, IsFloat | IsUntyped, 0, "untyped float"},
	UntypedComplex: {UntypedComplex, IsComplex | IsUntyped, 0, "untyped complex"},
	UntypedString:  {UntypedString, IsString | IsUntyped, 0, "untyped string"},
	UntypedNil:     {UntypedNil, IsUntyped, 0, "untyped nil"},
}

var aliases = [...]*Basic{
	{Byte, IsInteger | IsUnsigned, 1, "byte"},
	{Rune, IsInteger, 4, "rune"},
}

var predeclaredConstants = [...]*Const{
	{"true", Typ[UntypedBool], true, nil},
	{"false", Typ[UntypedBool], false, nil},
	{"iota", Typ[UntypedInt], zeroConst, nil},
	{"nil", Typ[UntypedNil], nilConst, nil},
}

var predeclaredFunctions = [...]*builtin{
	{_Append, "append", 1, true, false},
	{_Cap, "cap", 1, false, false},
	{_Close, "close", 1, false, true},
	{_Complex, "complex", 2, false, false},
	{_Copy, "copy", 2, false, true},
	{_Delete, "delete", 2, false, true},
	{_Imag, "imag", 1, false, false},
	{_Len, "len", 1, false, false},
	{_Make, "make", 1, true, false},
	{_New, "new", 1, false, false},
	{_Panic, "panic", 1, false, true},
	{_Print, "print", 1, true, true},
	{_Println, "println", 1, true, true},
	{_Real, "real", 1, false, false},
	{_Recover, "recover", 0, false, true},

	{_Alignof, "Alignof", 1, false, false},
	{_Offsetof, "Offsetof", 1, false, false},
	{_Sizeof, "Sizeof", 1, false, false},
}

func init() {
	Universe = new(Scope)
	Unsafe = &Package{Name: "unsafe", Scope: new(Scope)}

	// predeclared types
	for _, t := range Typ {
		def(&TypeName{Name: t.Name, Type: t})
	}
	for _, t := range aliases {
		def(&TypeName{Name: t.Name, Type: t})
	}

	// error type
	{
		// Error has a nil package in its qualified name since it is in no package
		err := &Method{QualifiedName{nil, "Error"}, &Signature{Results: []*Var{{Name: "", Type: Typ[String]}}}}
		def(&TypeName{Name: "error", Type: &NamedType{Underlying: &Interface{Methods: []*Method{err}}}})
	}

	for _, c := range predeclaredConstants {
		def(c)
	}

	for _, f := range predeclaredFunctions {
		def(&Func{Name: f.name, Type: f})
	}

	universeIota = Universe.Lookup("iota").(*Const)
}

// Objects with names containing blanks are internal and not entered into
// a scope. Objects with exported names are inserted in the unsafe package
// scope; other objects are inserted in the universe scope.
//
func def(obj Object) {
	name := obj.GetName()
	if strings.Index(name, " ") >= 0 {
		return // nothing to do
	}
	// fix Obj link for named types
	if typ, ok := obj.GetType().(*NamedType); ok {
		typ.Obj = obj.(*TypeName)
	}
	// exported identifiers go into package unsafe
	scope := Universe
	if ast.IsExported(name) {
		scope = Unsafe.Scope
	}
	if scope.Insert(obj) != nil {
		panic("internal error: double declaration")
	}
}
