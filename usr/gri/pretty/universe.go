// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Universe

import (
	"array";
	Globals "globals";
	Object "object";
	Type "type";
)


export var (
	scope *Globals.Scope;
	types array.Array;
	
	// internal types
	void_typ,
	bad_typ,
	nil_typ,
	
	// basic types
	bool_typ,
	uint8_typ,
	uint16_typ,
	uint32_typ,
	uint64_typ,
	int8_typ,
	int16_typ,
	int32_typ,
	int64_typ,
	float32_typ,
	float64_typ,
	float80_typ,
	string_typ,
	integer_typ,
	
	// convenience types
	byte_typ,
	uint_typ,
	int_typ,
	float_typ,
	uintptr_typ *Globals.Type;
	
	true_obj,
	false_obj,
	iota_obj,
	nil_obj *Globals.Object;
)


func DeclObj(kind int, ident string, typ *Globals.Type) *Globals.Object {
	obj := Globals.NewObject(-1 /* no source pos */, kind, ident);
	obj.typ = typ;
	if kind == Object.TYPE && typ.obj == nil {
		typ.obj = obj;  // set primary type object
	}
	scope.Insert(obj);
	return obj
}


func DeclType(form int, ident string, size int) *Globals.Type {
  typ := Globals.NewType(form);
  typ.size = size;
  return DeclObj(Object.TYPE, ident, typ).typ;
}


func Register(typ *Globals.Type) *Globals.Type {
	typ.ref = types.Len();
	types.Push(typ);
	return typ;
}


func init() {
	scope = Globals.NewScope(nil);  // universe has no parent
	types.Init(32);
	
	// Interal types
	void_typ = Globals.NewType(Type.VOID);
	Globals.Universe_void_typ = void_typ;
	bad_typ = Globals.NewType(Type.BAD);
	nil_typ = Globals.NewType(Type.NIL);
	
	// Basic types
	bool_typ = Register(DeclType(Type.BOOL, "bool", 1));
	uint8_typ = Register(DeclType(Type.UINT, "uint8", 1));
	uint16_typ = Register(DeclType(Type.UINT, "uint16", 2));
	uint32_typ = Register(DeclType(Type.UINT, "uint32", 4));
	uint64_typ = Register(DeclType(Type.UINT, "uint64", 8));
	int8_typ = Register(DeclType(Type.INT, "int8", 1));
	int16_typ = Register(DeclType(Type.INT, "int16", 2));
	int32_typ = Register(DeclType(Type.INT, "int32", 4));
	int64_typ = Register(DeclType(Type.INT, "int64", 8));
	float32_typ = Register(DeclType(Type.FLOAT, "float32", 4));
	float64_typ = Register(DeclType(Type.FLOAT, "float64", 8));
	float80_typ = Register(DeclType(Type.FLOAT, "float80", 10));
	string_typ = Register(DeclType(Type.STRING, "string", 8));
	integer_typ = Register(DeclType(Type.INTEGER, "integer", 8));

	// All but 'byte' should be platform-dependent, eventually.
	byte_typ = Register(DeclType(Type.UINT, "byte", 1));
	uint_typ = Register(DeclType(Type.UINT, "uint", 4));
	int_typ = Register(DeclType(Type.INT, "int", 4));
	float_typ = Register(DeclType(Type.FLOAT, "float", 4));
	uintptr_typ = Register(DeclType(Type.UINT, "uintptr", 8));

	// Predeclared constants
	true_obj = DeclObj(Object.CONST, "true", bool_typ);
	false_obj = DeclObj(Object.CONST, "false", bool_typ);
	iota_obj = DeclObj(Object.CONST, "iota", int_typ);
	nil_obj = DeclObj(Object.CONST, "nil", nil_typ);

	// Builtin functions
	DeclObj(Object.BUILTIN, "len", void_typ);
	DeclObj(Object.BUILTIN, "new", void_typ);
	DeclObj(Object.BUILTIN, "panic", void_typ);
	DeclObj(Object.BUILTIN, "print", void_typ);
	
	// scope.Print();
}
