// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package SymbolTable

import (
	"container/vector";
	"unicode";
	"utf8";
)


type Type struct;


// ----------------------------------------------------------------------------
// Support

func assert(pred bool) {
	if !pred {
		panic("assertion failed");
	}
}


// ----------------------------------------------------------------------------
// Objects

// Object represents a language object, such as a constant, variable, type, etc.

const /* kind */ (
	BADOBJ = iota;  // error handling
	NONE;  // kind unknown
	CONST; TYPE; VAR; FIELD; FUNC; BUILTIN; PACKAGE; LABEL;
	END;  // end of scope (import/export only)
)


func KindStr(kind int) string {
	switch kind {
	case BADOBJ: return "BADOBJ";
	case NONE: return "NONE";
	case CONST: return "CONST";
	case TYPE: return "TYPE";
	case VAR: return "VAR";
	case FIELD: return "FIELD";
	case FUNC: return "FUNC";
	case BUILTIN: return "BUILTIN";
	case PACKAGE: return "PACKAGE";
	case LABEL: return "LABEL";
	case END: return "END";
	}
	return "<unknown Object kind>";
}


type Object struct {
	Id int;  // unique id

	Pos int;  // source position (< 0 if unknown position)
	Kind int;  // object kind
	Ident string;
	Typ *Type;  // nil for packages
	Pnolev int;  // >= 0: package no., <= 0: function nesting level, 0: global level
}


func (obj *Object) IsExported() bool {
	switch obj.Kind {
	case NONE /* FUNC for now */, CONST, TYPE, VAR, FUNC:
		ch, size := utf8.DecodeRuneInString(obj.Ident,  0);
		return unicode.IsUpper(ch);
	}
	return false;
}


func (obj* Object) String() string {
	if obj != nil {
		return
			"Object(" +
			KindStr(obj.Kind) + ", " +
			obj.Ident +
			")";
	}
	return "nil";
}


var Universe_void_typ *Type  // initialized by Universe to Universe.void_typ
var objectId int;

func NewObject(pos, kind int, ident string) *Object {
	obj := new(Object);
	obj.Id = objectId;
	objectId++;

	obj.Pos = pos;
	obj.Kind = kind;
	obj.Ident = ident;
	obj.Typ = Universe_void_typ;  // TODO would it be better to use nil instead?
	obj.Pnolev = 0;

	return obj;
}


// ----------------------------------------------------------------------------
// Scopes

type Scope struct {
	Parent *Scope;
	entries map[string] *Object;
}


func NewScope(parent *Scope) *Scope {
	scope := new(Scope);
	scope.Parent = parent;
	scope.entries = make(map[string] *Object, 8);
	return scope;
}


func (scope *Scope) LookupLocal(ident string) *Object {
	obj, found := scope.entries[ident];
	if found {
		return obj;
	}
	return nil;
}


func (scope *Scope) Lookup(ident string) *Object {
	for scope != nil {
		obj := scope.LookupLocal(ident);
		if obj != nil {
			return obj;
		}
		scope = scope.Parent;
	}
	return nil;
}


func (scope *Scope) add(obj *Object) {
	scope.entries[obj.Ident] = obj;
}


func (scope *Scope) Insert(obj *Object) {
	if scope.LookupLocal(obj.Ident) != nil {
		panic("obj already inserted");
	}
	scope.add(obj);
}


func (scope *Scope) InsertImport(obj *Object) *Object {
	 p := scope.LookupLocal(obj.Ident);
	 if p == nil {
		scope.add(obj);
		p = obj;
	 }
	 return p;
}


func (scope *Scope) Print() {
	print("scope {");
	for key := range scope.entries {
		print("\n  ", key);
	}
	print("\n}\n");
}


// ----------------------------------------------------------------------------
// Types

const /* form */ (
	// internal types
	// We should never see one of these.
	UNDEF = iota;

	// VOID types are used when we don't have a type. Never exported.
	// (exported type forms must be > 0)
	VOID;

	// BADTYPE types are compatible with any type and don't cause further errors.
	// They are introduced only as a result of an error in the source code. A
	// correct program cannot have BAD types.
	BADTYPE;

	// FORWARD types are forward-declared (incomplete) types. They can only
	// be used as element types of pointer types and must be resolved before
	// their internals are accessible.
	FORWARD;

	// TUPLE types represent multi-valued result types of functions and
	// methods.
	TUPLE;

	// The type of nil.
	NIL;

	// A type name
	TYPENAME;

	// basic types
	BOOL; UINT; INT; FLOAT; STRING; INTEGER;

	// composite types
	ALIAS; ARRAY; STRUCT; INTERFACE; MAP; CHANNEL; FUNCTION; METHOD; POINTER;

	// open-ended parameter type
	ELLIPSIS
)


func FormStr(form int) string {
	switch form {
	case VOID: return "VOID";
	case BADTYPE: return "BADTYPE";
	case FORWARD: return "FORWARD";
	case TUPLE: return "TUPLE";
	case NIL: return "NIL";
	case TYPENAME: return "TYPENAME";
	case BOOL: return "BOOL";
	case UINT: return "UINT";
	case INT: return "INT";
	case FLOAT: return "FLOAT";
	case STRING: return "STRING";
	case ALIAS: return "ALIAS";
	case ARRAY: return "ARRAY";
	case STRUCT: return "STRUCT";
	case INTERFACE: return "INTERFACE";
	case MAP: return "MAP";
	case CHANNEL: return "CHANNEL";
	case FUNCTION: return "FUNCTION";
	case METHOD: return "METHOD";
	case POINTER: return "POINTER";
	case ELLIPSIS: return "ELLIPSIS";
	}
	return "<unknown Type form>";
}


const /* channel mode */ (
	FULL = iota;
	SEND;
	RECV;
)


type Type struct {
	Id int;  // unique id

	Ref int;  // for exporting only: >= 0 means already exported
	Form int;  // type form
	Size int;  // size in bytes
	Obj *Object;  // primary type object or nil
	Scope *Scope;  // locals, fields & methods

	// syntactic components
	Pos int;  // source position (< 0 if unknown position)
	Len int;  // array length
	Mode int;  // channel mode
	Key *Type;  // receiver type or map key
	Elt *Type;  // type name type, array, map, channel or pointer element type, function result type
	List *vector.Vector; End int;  // struct fields, interface methods, function parameters
}


var typeId int;

func NewType(pos, form int) *Type {
	typ := new(Type);
	typ.Id = typeId;
	typeId++;

	typ.Ref = -1;  // not yet exported
	typ.Pos = pos;
	typ.Form = form;

	return typ;
}


func (typ* Type) String() string {
	if typ != nil {
		return
			"Type(" +
			FormStr(typ.Form) +
			")";
	}
	return "nil";
}


// ----------------------------------------------------------------------------
// Universe scope

var (
	Universe *Scope;
	PredeclaredTypes vector.Vector;

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
	Uintptr_typ *Type;

	True_obj,
	False_obj,
	Iota_obj,
	Nil_obj *Object;
)


func declObj(kind int, ident string, typ *Type) *Object {
	obj := NewObject(-1 /* no source pos */, kind, ident);
	obj.Typ = typ;
	if kind == TYPE && typ.Obj == nil {
		typ.Obj = obj;  // set primary type object
	}
	Universe.Insert(obj);
	return obj
}


func declType(form int, ident string, size int) *Type {
  typ := NewType(-1 /* no source pos */, form);
  typ.Size = size;
  return declObj(TYPE, ident, typ).Typ;
}


func register(typ *Type) *Type {
	typ.Ref = PredeclaredTypes.Len();
	PredeclaredTypes.Push(typ);
	return typ;
}


func init() {
	Universe = NewScope(nil);  // universe has no parent
	PredeclaredTypes.Init(32);

	// Interal types
	Void_typ = NewType(-1 /* no source pos */, VOID);
	Universe_void_typ = Void_typ;
	Bad_typ = NewType(-1 /* no source pos */, BADTYPE);
	Nil_typ = NewType(-1 /* no source pos */, NIL);

	// Basic types
	Bool_typ = register(declType(BOOL, "bool", 1));
	Uint8_typ = register(declType(UINT, "uint8", 1));
	Uint16_typ = register(declType(UINT, "uint16", 2));
	Uint32_typ = register(declType(UINT, "uint32", 4));
	Uint64_typ = register(declType(UINT, "uint64", 8));
	Int8_typ = register(declType(INT, "int8", 1));
	Int16_typ = register(declType(INT, "int16", 2));
	Int32_typ = register(declType(INT, "int32", 4));
	Int64_typ = register(declType(INT, "int64", 8));
	Float32_typ = register(declType(FLOAT, "float32", 4));
	Float64_typ = register(declType(FLOAT, "float64", 8));
	Float80_typ = register(declType(FLOAT, "float80", 10));
	String_typ = register(declType(STRING, "string", 8));
	Integer_typ = register(declType(INTEGER, "integer", 8));

	// All but 'byte' should be platform-dependent, eventually.
	Byte_typ = register(declType(UINT, "byte", 1));
	Uint_typ = register(declType(UINT, "uint", 4));
	Int_typ = register(declType(INT, "int", 4));
	Float_typ = register(declType(FLOAT, "float", 4));
	Uintptr_typ = register(declType(UINT, "uintptr", 8));

	// Predeclared constants
	True_obj = declObj(CONST, "true", Bool_typ);
	False_obj = declObj(CONST, "false", Bool_typ);
	Iota_obj = declObj(CONST, "iota", Int_typ);
	Nil_obj = declObj(CONST, "nil", Nil_typ);

	// Builtin functions
	declObj(BUILTIN, "len", Void_typ);
	declObj(BUILTIN, "new", Void_typ);
	declObj(BUILTIN, "panic", Void_typ);
	declObj(BUILTIN, "print", Void_typ);

	// scope.Print();
}
