// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements the universe and unsafe package scopes.

package types

import (
	"go/token"
	"strings"

	"code.google.com/p/go.tools/go/exact"
)

var (
	Universe     *Scope
	Unsafe       *Package
	universeIota *Const
)

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

func defPredeclaredTypes() {
	for _, t := range Typ {
		def(NewTypeName(token.NoPos, nil, t.name, t))
	}
	for _, t := range aliases {
		def(NewTypeName(token.NoPos, nil, t.name, t))
	}

	// Error has a nil package in its qualified name since it is in no package
	res := NewVar(token.NoPos, nil, "", Typ[String])
	sig := &Signature{results: NewTuple(res)}
	err := NewFunc(token.NoPos, nil, "Error", sig)
	typ := &Named{underlying: NewInterface([]*Func{err}, nil), complete: true}
	sig.recv = NewVar(token.NoPos, nil, "", typ)
	def(NewTypeName(token.NoPos, nil, "error", typ))
}

var predeclaredConsts = [...]struct {
	name string
	kind BasicKind
	val  exact.Value
}{
	{"true", UntypedBool, exact.MakeBool(true)},
	{"false", UntypedBool, exact.MakeBool(false)},
	{"iota", UntypedInt, exact.MakeInt64(0)},
}

func defPredeclaredConsts() {
	for _, c := range predeclaredConsts {
		def(NewConst(token.NoPos, nil, c.name, Typ[c.kind], c.val))
	}
}

func defPredeclaredNil() {
	def(&Nil{object{name: "nil", typ: Typ[UntypedNil]}})
}

// A builtinId is the id of a builtin function.
type builtinId int

const (
	// universe scope
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

	// package unsafe
	_Alignof
	_Offsetof
	_Sizeof

	// testing support
	_Assert
	_Trace
)

var predeclaredFuncs = [...]struct {
	name     string
	nargs    int
	variadic bool
	kind     exprKind
}{
	_Append:  {"append", 1, true, expression},
	_Cap:     {"cap", 1, false, expression},
	_Close:   {"close", 1, false, statement},
	_Complex: {"complex", 2, false, expression},
	_Copy:    {"copy", 2, false, statement},
	_Delete:  {"delete", 2, false, statement},
	_Imag:    {"imag", 1, false, expression},
	_Len:     {"len", 1, false, expression},
	_Make:    {"make", 1, true, expression},
	_New:     {"new", 1, false, expression},
	_Panic:   {"panic", 1, false, statement},
	_Print:   {"print", 0, true, statement},
	_Println: {"println", 0, true, statement},
	_Real:    {"real", 1, false, expression},
	_Recover: {"recover", 0, false, statement},

	_Alignof:  {"Alignof", 1, false, expression},
	_Offsetof: {"Offsetof", 1, false, expression},
	_Sizeof:   {"Sizeof", 1, false, expression},

	_Assert: {"assert", 1, false, statement},
	_Trace:  {"trace", 0, true, statement},
}

func defPredeclaredFuncs() {
	for i := range predeclaredFuncs {
		id := builtinId(i)
		if id == _Assert || id == _Trace {
			continue // only define these in testing environment
		}
		def(newBuiltin(id))
	}
}

// DefPredeclaredTestFuncs defines the assert and trace built-ins.
// These built-ins are intended for debugging and testing of this
// package only.
func DefPredeclaredTestFuncs() {
	if Universe.Lookup("assert") != nil {
		return // already defined
	}
	def(newBuiltin(_Assert))
	def(newBuiltin(_Trace))
}

func init() {
	Universe = NewScope(nil)
	Unsafe = NewPackage("unsafe", "unsafe", NewScope(Universe))
	Unsafe.complete = true

	defPredeclaredTypes()
	defPredeclaredConsts()
	defPredeclaredNil()
	defPredeclaredFuncs()

	universeIota = Universe.Lookup("iota").(*Const)
}

// Objects with names containing blanks are internal and not entered into
// a scope. Objects with exported names are inserted in the unsafe package
// scope; other objects are inserted in the universe scope.
//
func def(obj Object) {
	name := obj.Name()
	if strings.Index(name, " ") >= 0 {
		return // nothing to do
	}
	// fix Obj link for named types
	if typ, ok := obj.Type().(*Named); ok {
		typ.obj = obj.(*TypeName)
	}
	// exported identifiers go into package unsafe
	scope := Universe
	if obj.IsExported() {
		scope = Unsafe.scope
		// set Pkg field
		switch obj := obj.(type) {
		case *TypeName:
			obj.pkg = Unsafe
		case *Builtin:
			obj.pkg = Unsafe
		default:
			unreachable()
		}
	}
	if scope.Insert(obj) != nil {
		panic("internal error: double declaration")
	}
}
