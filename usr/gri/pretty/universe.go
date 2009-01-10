// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Universe

import (
	"array";
	AST "ast";
)


export var (
	scope *AST.Scope;
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
	uintptr_typ *AST.Type;
	
	true_obj,
	false_obj,
	iota_obj,
	nil_obj *AST.Object;
)


func DeclObj(kind int, ident string, typ *AST.Type) *AST.Object {
	obj := AST.NewObject(-1 /* no source pos */, kind, ident);
	obj.typ = typ;
	if kind == AST.TYPE && typ.obj == nil {
		typ.obj = obj;  // set primary type object
	}
	scope.Insert(obj);
	return obj
}


func DeclType(form int, ident string, size int) *AST.Type {
  typ := AST.NewType(-1 /* no source pos */, form);
  typ.size = size;
  return DeclObj(AST.TYPE, ident, typ).typ;
}


func Register(typ *AST.Type) *AST.Type {
	typ.ref = types.Len();
	types.Push(typ);
	return typ;
}


func init() {
	scope = AST.NewScope(nil);  // universe has no parent
	types.Init(32);
	
	// Interal types
	void_typ = AST.NewType(-1 /* no source pos */, AST.VOID);
	AST.Universe_void_typ = void_typ;
	bad_typ = AST.NewType(-1 /* no source pos */, AST.BADTYPE);
	nil_typ = AST.NewType(-1 /* no source pos */, AST.NIL);
	
	// Basic types
	bool_typ = Register(DeclType(AST.BOOL, "bool", 1));
	uint8_typ = Register(DeclType(AST.UINT, "uint8", 1));
	uint16_typ = Register(DeclType(AST.UINT, "uint16", 2));
	uint32_typ = Register(DeclType(AST.UINT, "uint32", 4));
	uint64_typ = Register(DeclType(AST.UINT, "uint64", 8));
	int8_typ = Register(DeclType(AST.INT, "int8", 1));
	int16_typ = Register(DeclType(AST.INT, "int16", 2));
	int32_typ = Register(DeclType(AST.INT, "int32", 4));
	int64_typ = Register(DeclType(AST.INT, "int64", 8));
	float32_typ = Register(DeclType(AST.FLOAT, "float32", 4));
	float64_typ = Register(DeclType(AST.FLOAT, "float64", 8));
	float80_typ = Register(DeclType(AST.FLOAT, "float80", 10));
	string_typ = Register(DeclType(AST.STRING, "string", 8));
	integer_typ = Register(DeclType(AST.INTEGER, "integer", 8));

	// All but 'byte' should be platform-dependent, eventually.
	byte_typ = Register(DeclType(AST.UINT, "byte", 1));
	uint_typ = Register(DeclType(AST.UINT, "uint", 4));
	int_typ = Register(DeclType(AST.INT, "int", 4));
	float_typ = Register(DeclType(AST.FLOAT, "float", 4));
	uintptr_typ = Register(DeclType(AST.UINT, "uintptr", 8));

	// Predeclared constants
	true_obj = DeclObj(AST.CONST, "true", bool_typ);
	false_obj = DeclObj(AST.CONST, "false", bool_typ);
	iota_obj = DeclObj(AST.CONST, "iota", int_typ);
	nil_obj = DeclObj(AST.CONST, "nil", nil_typ);

	// Builtin functions
	DeclObj(AST.BUILTIN, "len", void_typ);
	DeclObj(AST.BUILTIN, "new", void_typ);
	DeclObj(AST.BUILTIN, "panic", void_typ);
	DeclObj(AST.BUILTIN, "print", void_typ);
	
	// scope.Print();
}
