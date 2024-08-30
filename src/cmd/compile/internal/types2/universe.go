// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file sets up the universe scope and the unsafe package.

package types2

import (
	"go/constant"
	"strings"
)

// The Universe scope contains all predeclared objects of Go.
// It is the outermost scope of any chain of nested scopes.
var Universe *Scope

// The Unsafe package is the package returned by an importer
// for the import path "unsafe".
var Unsafe *Package

var (
	universeIota       Object
	universeByte       Type // uint8 alias, but has name "byte"
	universeRune       Type // int32 alias, but has name "rune"
	universeAnyNoAlias *TypeName
	universeAnyAlias   *TypeName
	universeError      Type
	universeComparable Object
)

// Typ contains the predeclared *Basic types indexed by their
// corresponding BasicKind.
//
// The *Basic type for Typ[Byte] will have the name "uint8".
// Use Universe.Lookup("byte").Type() to obtain the specific
// alias basic type named "byte" (and analogous for "rune").
var Typ = [...]*Basic{
	Invalid: {Invalid, 0, "invalid type"},

	Bool:          {Bool, IsBoolean, "bool"},
	Int:           {Int, IsInteger, "int"},
	Int8:          {Int8, IsInteger, "int8"},
	Int16:         {Int16, IsInteger, "int16"},
	Int32:         {Int32, IsInteger, "int32"},
	Int64:         {Int64, IsInteger, "int64"},
	Uint:          {Uint, IsInteger | IsUnsigned, "uint"},
	Uint8:         {Uint8, IsInteger | IsUnsigned, "uint8"},
	Uint16:        {Uint16, IsInteger | IsUnsigned, "uint16"},
	Uint32:        {Uint32, IsInteger | IsUnsigned, "uint32"},
	Uint64:        {Uint64, IsInteger | IsUnsigned, "uint64"},
	Uintptr:       {Uintptr, IsInteger | IsUnsigned, "uintptr"},
	Float32:       {Float32, IsFloat, "float32"},
	Float64:       {Float64, IsFloat, "float64"},
	Complex64:     {Complex64, IsComplex, "complex64"},
	Complex128:    {Complex128, IsComplex, "complex128"},
	String:        {String, IsString, "string"},
	UnsafePointer: {UnsafePointer, 0, "Pointer"},

	UntypedBool:    {UntypedBool, IsBoolean | IsUntyped, "untyped bool"},
	UntypedInt:     {UntypedInt, IsInteger | IsUntyped, "untyped int"},
	UntypedRune:    {UntypedRune, IsInteger | IsUntyped, "untyped rune"},
	UntypedFloat:   {UntypedFloat, IsFloat | IsUntyped, "untyped float"},
	UntypedComplex: {UntypedComplex, IsComplex | IsUntyped, "untyped complex"},
	UntypedString:  {UntypedString, IsString | IsUntyped, "untyped string"},
	UntypedNil:     {UntypedNil, IsUntyped, "untyped nil"},
}

var basicAliases = [...]*Basic{
	{Byte, IsInteger | IsUnsigned, "byte"},
	{Rune, IsInteger, "rune"},
}

func defPredeclaredTypes() {
	for _, t := range Typ {
		def(NewTypeName(nopos, nil, t.name, t))
	}
	for _, t := range basicAliases {
		def(NewTypeName(nopos, nil, t.name, t))
	}

	// type any = interface{}
	//
	// Implement two representations of any: one for the legacy gotypesalias=0,
	// and one for gotypesalias=1. This is necessary for consistent
	// representation of interface aliases during type checking, and is
	// implemented via hijacking [Scope.Lookup] for the [Universe] scope.
	//
	// Both representations use the same distinguished pointer for their RHS
	// interface type, allowing us to detect any (even with the legacy
	// representation), and format it as "any" rather than interface{}, which
	// clarifies user-facing error messages significantly.
	//
	// TODO(rfindley): once the gotypesalias GODEBUG variable is obsolete (and we
	// consistently use the Alias node), we should be able to clarify user facing
	// error messages without using a distinguished pointer for the any
	// interface.
	{
		universeAnyNoAlias = NewTypeName(nopos, nil, "any", &Interface{complete: true, tset: &topTypeSet})
		universeAnyNoAlias.setColor(black)
		// ensure that the any TypeName reports a consistent Parent, after
		// hijacking Universe.Lookup with gotypesalias=0.
		universeAnyNoAlias.setParent(Universe)

		// It shouldn't matter which representation of any is actually inserted
		// into the Universe, but we lean toward the future and insert the Alias
		// representation.
		universeAnyAlias = NewTypeName(nopos, nil, "any", nil)
		universeAnyAlias.setColor(black)
		_ = NewAlias(universeAnyAlias, universeAnyNoAlias.Type().Underlying()) // Link TypeName and Alias
		def(universeAnyAlias)
	}

	// type error interface{ Error() string }
	{
		obj := NewTypeName(nopos, nil, "error", nil)
		obj.setColor(black)
		typ := NewNamed(obj, nil, nil)

		// error.Error() string
		recv := NewVar(nopos, nil, "", typ)
		res := NewVar(nopos, nil, "", Typ[String])
		sig := NewSignatureType(recv, nil, nil, nil, NewTuple(res), false)
		err := NewFunc(nopos, nil, "Error", sig)

		// interface{ Error() string }
		ityp := &Interface{methods: []*Func{err}, complete: true}
		computeInterfaceTypeSet(nil, nopos, ityp) // prevent races due to lazy computation of tset

		typ.SetUnderlying(ityp)
		def(obj)
	}

	// type comparable interface{} // marked as comparable
	{
		obj := NewTypeName(nopos, nil, "comparable", nil)
		obj.setColor(black)
		typ := NewNamed(obj, nil, nil)

		// interface{} // marked as comparable
		ityp := &Interface{complete: true, tset: &_TypeSet{nil, allTermlist, true}}

		typ.SetUnderlying(ityp)
		def(obj)
	}
}

var predeclaredConsts = [...]struct {
	name string
	kind BasicKind
	val  constant.Value
}{
	{"true", UntypedBool, constant.MakeBool(true)},
	{"false", UntypedBool, constant.MakeBool(false)},
	{"iota", UntypedInt, constant.MakeInt64(0)},
}

func defPredeclaredConsts() {
	for _, c := range predeclaredConsts {
		def(NewConst(nopos, nil, c.name, Typ[c.kind], c.val))
	}
}

func defPredeclaredNil() {
	def(&Nil{object{name: "nil", typ: Typ[UntypedNil], color_: black}})
}

// A builtinId is the id of a builtin function.
type builtinId int

const (
	// universe scope
	_Append builtinId = iota
	_Cap
	_Clear
	_Close
	_Complex
	_Copy
	_Delete
	_Imag
	_Len
	_Make
	_Max
	_Min
	_New
	_Panic
	_Print
	_Println
	_Real
	_Recover

	// package unsafe
	_Add
	_Alignof
	_Offsetof
	_Sizeof
	_Slice
	_SliceData
	_String
	_StringData

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
	_Clear:   {"clear", 1, false, statement},
	_Close:   {"close", 1, false, statement},
	_Complex: {"complex", 2, false, expression},
	_Copy:    {"copy", 2, false, statement},
	_Delete:  {"delete", 2, false, statement},
	_Imag:    {"imag", 1, false, expression},
	_Len:     {"len", 1, false, expression},
	_Make:    {"make", 1, true, expression},
	// To disable max/min, remove the next two lines.
	_Max:     {"max", 1, true, expression},
	_Min:     {"min", 1, true, expression},
	_New:     {"new", 1, false, expression},
	_Panic:   {"panic", 1, false, statement},
	_Print:   {"print", 0, true, statement},
	_Println: {"println", 0, true, statement},
	_Real:    {"real", 1, false, expression},
	_Recover: {"recover", 0, false, statement},

	_Add:        {"Add", 2, false, expression},
	_Alignof:    {"Alignof", 1, false, expression},
	_Offsetof:   {"Offsetof", 1, false, expression},
	_Sizeof:     {"Sizeof", 1, false, expression},
	_Slice:      {"Slice", 2, false, expression},
	_SliceData:  {"SliceData", 1, false, expression},
	_String:     {"String", 2, false, expression},
	_StringData: {"StringData", 1, false, expression},

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
	Universe = NewScope(nil, nopos, nopos, "universe")
	Unsafe = NewPackage("unsafe", "unsafe")
	Unsafe.complete = true

	defPredeclaredTypes()
	defPredeclaredConsts()
	defPredeclaredNil()
	defPredeclaredFuncs()

	universeIota = Universe.Lookup("iota")
	universeByte = Universe.Lookup("byte").Type()
	universeRune = Universe.Lookup("rune").Type()
	universeError = Universe.Lookup("error").Type()
	universeComparable = Universe.Lookup("comparable")
}

// Objects with names containing blanks are internal and not entered into
// a scope. Objects with exported names are inserted in the unsafe package
// scope; other objects are inserted in the universe scope.
func def(obj Object) {
	assert(obj.color() == black)
	name := obj.Name()
	if strings.Contains(name, " ") {
		return // nothing to do
	}
	// fix Obj link for named types
	if typ := asNamed(obj.Type()); typ != nil {
		typ.obj = obj.(*TypeName)
	}
	// exported identifiers go into package unsafe
	scope := Universe
	if obj.Exported() {
		scope = Unsafe.scope
		// set Pkg field
		switch obj := obj.(type) {
		case *TypeName:
			obj.pkg = Unsafe
		case *Builtin:
			obj.pkg = Unsafe
		default:
			panic("unreachable")
		}
	}
	if scope.Insert(obj) != nil {
		panic("double declaration of predeclared identifier")
	}
}
