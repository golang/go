// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Universe

import (
	"array";
	AST "ast";
)


export var (
	Scope *AST.Scope;
	Types array.Array;
	
	// internal types
	Void_typ,
	Bad_typ,
	Nil_typ,
	
	// basic types
	Bool_typ,
	Uint8_typ,
	Uint16_typ,
	Uint32_typ,
	Uint64_typ,
	Int8_typ,
	Int16_typ,
	Int32_typ,
	Int64_typ,
	Float32_typ,
	Float64_typ,
	Float80_typ,
	String_typ,
	Integer_typ,
	
	// convenience types
	Byte_typ,
	Uint_typ,
	Int_typ,
	Float_typ,
	Uintptr_typ *AST.Type;
	
	True_obj,
	False_obj,
	Iota_obj,
	Nil_obj *AST.Object;
)


func declObj(kind int, ident string, typ *AST.Type) *AST.Object {
	obj := AST.NewObject(-1 /* no source pos */, kind, ident);
	obj.Typ = typ;
	if kind == AST.TYPE && typ.Obj == nil {
		typ.Obj = obj;  // set primary type object
	}
	Scope.Insert(obj);
	return obj
}


func declType(form int, ident string, size int) *AST.Type {
  typ := AST.NewType(-1 /* no source pos */, form);
  typ.Size = size;
  return declObj(AST.TYPE, ident, typ).Typ;
}


func register(typ *AST.Type) *AST.Type {
	typ.Ref = Types.Len();
	Types.Push(typ);
	return typ;
}


func init() {
	Scope = AST.NewScope(nil);  // universe has no parent
	Types.Init(32);
	
	// Interal types
	Void_typ = AST.NewType(-1 /* no source pos */, AST.VOID);
	AST.Universe_void_typ = Void_typ;
	Bad_typ = AST.NewType(-1 /* no source pos */, AST.BADTYPE);
	Nil_typ = AST.NewType(-1 /* no source pos */, AST.NIL);
	
	// Basic types
	Bool_typ = register(declType(AST.BOOL, "bool", 1));
	Uint8_typ = register(declType(AST.UINT, "uint8", 1));
	Uint16_typ = register(declType(AST.UINT, "uint16", 2));
	Uint32_typ = register(declType(AST.UINT, "uint32", 4));
	Uint64_typ = register(declType(AST.UINT, "uint64", 8));
	Int8_typ = register(declType(AST.INT, "int8", 1));
	Int16_typ = register(declType(AST.INT, "int16", 2));
	Int32_typ = register(declType(AST.INT, "int32", 4));
	Int64_typ = register(declType(AST.INT, "int64", 8));
	Float32_typ = register(declType(AST.FLOAT, "float32", 4));
	Float64_typ = register(declType(AST.FLOAT, "float64", 8));
	Float80_typ = register(declType(AST.FLOAT, "float80", 10));
	String_typ = register(declType(AST.STRING, "string", 8));
	Integer_typ = register(declType(AST.INTEGER, "integer", 8));

	// All but 'byte' should be platform-dependent, eventually.
	Byte_typ = register(declType(AST.UINT, "byte", 1));
	Uint_typ = register(declType(AST.UINT, "uint", 4));
	Int_typ = register(declType(AST.INT, "int", 4));
	Float_typ = register(declType(AST.FLOAT, "float", 4));
	Uintptr_typ = register(declType(AST.UINT, "uintptr", 8));

	// Predeclared constants
	True_obj = declObj(AST.CONST, "true", Bool_typ);
	False_obj = declObj(AST.CONST, "false", Bool_typ);
	Iota_obj = declObj(AST.CONST, "iota", Int_typ);
	Nil_obj = declObj(AST.CONST, "nil", Nil_typ);

	// Builtin functions
	declObj(AST.BUILTIN, "len", Void_typ);
	declObj(AST.BUILTIN, "new", Void_typ);
	declObj(AST.BUILTIN, "panic", Void_typ);
	declObj(AST.BUILTIN, "print", Void_typ);
	
	// scope.Print();
}
