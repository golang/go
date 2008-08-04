// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Universe

import Globals "globals"
import Object "object"
import Type "type"


export var (
	scope *Globals.Scope;
	types *Globals.List;
	
	// internal types
	undef_t,
	bad_t,
	nil_t,
	
	// basic types
	bool_t,
	uint8_t,
	uint16_t,
	uint32_t,
	uint64_t,
	int8_t,
	int16_t,
	int32_t,
	int64_t,
	float32_t,
	float64_t,
	float80_t,
	string_t,
	any_t,
	
	// alias types
	byte_t,
	ushort_t,
	uint_t,
	ulong_t,
	short_t,
	int_t,
	long_t,
	float_t,
	double_t,
	ptrint_t *Globals.Type;
	
	true_,
	false_ *Globals.Object;
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


func DeclAlias(ident string, typ *Globals.Type) *Globals.Type {
	alias := Globals.NewType(Type.ALIAS);
	alias.elt = typ;
	return DeclObj(Object.TYPE, ident, alias).typ;
}


func Register(typ *Globals.Type) *Globals.Type {
	if types.len_ < 0 {
		panic "types.len_ < 0";
	}
	typ.ref = types.len_;
	types.AddTyp(typ);
	return typ;
}


func init() {
	scope = Globals.NewScope(nil);  // universe has no parent
	types = Globals.NewList();
	
	// Interal types
	undef_t = Globals.NewType(Type.UNDEF);
	Globals.Universe_undef_t = undef_t;
	bad_t = Globals.NewType(Type.BAD);
	nil_t = DeclType(Type.NIL, "nil", 8);
	
	// Basic types
	bool_t = Register(DeclType(Type.BOOL, "bool", 1));
	uint8_t = Register(DeclType(Type.UINT, "uint8", 1));
	uint16_t = Register(DeclType(Type.UINT, "uint16", 2));
	uint32_t = Register(DeclType(Type.UINT, "uint32", 4));
	uint64_t = Register(DeclType(Type.UINT, "uint64", 8));
	int8_t = Register(DeclType(Type.INT, "int8", 1));
	int16_t = Register(DeclType(Type.INT, "int16", 2));
	int32_t = Register(DeclType(Type.INT, "int32", 4));
	int64_t = Register(DeclType(Type.INT, "int64", 8));
	float32_t = Register(DeclType(Type.FLOAT, "float32", 4));
	float64_t = Register(DeclType(Type.FLOAT, "float64", 8));
	float80_t = Register(DeclType(Type.FLOAT, "float80", 10));
	string_t = Register(DeclType(Type.STRING, "string", 8));
	any_t = Register(DeclType(Type.ANY, "any", 8));

	// All but 'byte' should be platform-dependent, eventually.
	byte_t = Register(DeclAlias("byte", uint8_t));
	ushort_t = Register(DeclAlias("ushort", uint16_t));
	uint_t = Register(DeclAlias("uint", uint32_t));
	ulong_t = Register(DeclAlias("ulong", uint32_t));
	short_t = Register(DeclAlias("short", int16_t));
	int_t = Register(DeclAlias("int", int32_t));
	long_t = Register(DeclAlias("long", int32_t));
	float_t = Register(DeclAlias("float", float32_t));
	double_t = Register(DeclAlias("double", float64_t));
	ptrint_t = Register(DeclAlias("ptrint", uint64_t));

	// Predeclared constants
	true_ = DeclObj(Object.CONST, "true", bool_t);
	false_ = DeclObj(Object.CONST, "false", bool_t);

	// Builtin functions
	DeclObj(Object.FUNC, "len", Globals.NewType(Type.FUNCTION));  // incomplete
	
	// scope.Print();
}
