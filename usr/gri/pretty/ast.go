// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package AST

import (
	"array";
	"utf8";
	"unicode";
	Scanner "scanner";
)


export type (
	Object struct;
	Type struct;

	Expr struct;
	Stat struct;
	Decl struct;
)


// ----------------------------------------------------------------------------
// Objects

// Object represents a language object, such as a constant, variable, type, etc.

export const /* kind */ (
	BADOBJ = iota;  // error handling
	NONE;  // kind unknown
	CONST; TYPE; VAR; FIELD; FUNC; BUILTIN; PACKAGE; LABEL;
	END;  // end of scope (import/export only)
)


export func KindStr(kind int) string {
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


export type Object struct {
	Id int;  // unique id

	Pos int;  // source position (< 0 if unknown position)
	Kind int;  // object kind
	Ident string;
	Typ *Type;  // nil for packages
	Pnolev int;  // >= 0: package no., <= 0: function nesting level, 0: global level
	
	// attached values
	Block *array.Array; End int;  // stats for function literals; end of block pos
}


func (obj *Object) IsExported() bool {
	switch obj.Kind {
	case NONE /* FUNC for now */, CONST, TYPE, VAR, FUNC:
		ch, size := utf8.DecodeRuneInString(obj.Ident,  0);
		return unicode.IsUpper(ch);
	}
	return false;
}


export var Universe_void_typ *Type  // initialized by Universe to Universe.void_typ
var objectId int;

export func NewObject(pos, kind int, ident string) *Object {
	obj := new(Object);
	obj.Id = objectId;
	objectId++;
	
	obj.Pos = pos;
	obj.Kind = kind;
	obj.Ident = ident;
	obj.Typ = Universe_void_typ;
	obj.Pnolev = 0;

	return obj;
}


// ----------------------------------------------------------------------------
// Scopes

export type Scope struct {
	parent *Scope;
	entries map[string] *Object;
}


export func NewScope(parent *Scope) *Scope {
	scope := new(Scope);
	scope.parent = parent;
	scope.entries = make(map[string]*Object, 8);
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
		scope = scope.parent;
	}
	return nil;
}


func (scope *Scope) add(obj* Object) {
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
// All nodes have a source position and and token.

export type Node struct {
	Pos int;  // source position (< 0 => unknown position)
	Tok int;  // identifying token
}


// ----------------------------------------------------------------------------
// Expressions

export type Expr struct {
	Node;
	X, Y *Expr;  // binary (X, Y) and unary (Y) expressions
	Obj *Object;
}


func (x *Expr) Len() int {
	if x == nil {
		return 0;
	}
	n := 1;
	for ; x.Tok == Scanner.COMMA; x = x.Y {
		n++;
	}
	return n;
}


export func NewExpr(pos, tok int, x, y *Expr) *Expr {
	if x != nil && x.Tok == Scanner.TYPE || y != nil && y.Tok == Scanner.TYPE {
		panic("no type expression allowed");
	}
	e := new(Expr);
	e.Pos, e.Tok, e.X, e.Y = pos, tok, x, y;
	return e;
}


// TODO probably don't need the tok parameter eventually
export func NewLit(tok int, obj *Object) *Expr {
	e := new(Expr);
	e.Pos, e.Tok, e.Obj = obj.Pos, tok, obj;
	return e;
}


export var BadExpr = NewExpr(0, Scanner.ILLEGAL, nil, nil);


// ----------------------------------------------------------------------------
// Types

export const /* form */ (
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


export func FormStr(form int) string {
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


export const /* channel mode */ (
	FULL = iota;
	SEND;
	RECV;
)


export type Type struct {
	Id int;  // unique id

	Ref int;  // for exporting only: >= 0 means already exported
	Form int;  // type form
	Size int;  // size in bytes
	Obj *Object;  // primary type object or NULL
	Scope *Scope;  // forwards, structs, interfaces, functions

	// syntactic components
	Pos int;  // source position (< 0 if unknown position)
	Expr *Expr;  // type name, array length
	Mode int;  // channel mode
	Key *Type;  // receiver type or map key
	Elt *Type;  // array, map, channel or pointer element type, function result type
	List *array.Array; End int;  // struct fields, interface methods, function parameters
	Scope *Scope;  // struct fields, methods
}


var typeId int;

export func NewType(pos, form int) *Type {
	typ := new(Type);
	typ.Id = typeId;
	typeId++;

	typ.Ref = -1;  // not yet exported
	typ.Pos = pos;
	typ.Form = form;

	return typ;
}


func (t *Type) Nfields() int {
	if t.List == nil {
		return 0;
	}
	nx, nt := 0, 0;
	for i, n := 0, t.List.Len(); i < n; i++ {
		if t.List.At(i).(*Expr).Tok == Scanner.TYPE {
			nt++;
		} else {
			nx++;
		}
	}
	if nx == 0 {
		return nt;
	}
	return nx;
}


// requires complete Type.Pos access
export func NewTypeExpr(typ *Type) *Expr {
	obj := NewObject(typ.Pos, TYPE, "");
	obj.Typ = typ;
	return NewLit(Scanner.TYPE, obj);
}


export var BadType = NewType(0, Scanner.ILLEGAL);


// ----------------------------------------------------------------------------
// Statements

export type Stat struct {
	Node;
	Init, Post *Stat;
	Expr *Expr;
	Block *array.Array; End int;  // block end position
	Decl *Decl;
}


export func NewStat(pos, tok int) *Stat {
	s := new(Stat);
	s.Pos, s.Tok = pos, tok;
	return s;
}


export var BadStat = NewStat(0, Scanner.ILLEGAL);


// ----------------------------------------------------------------------------
// Declarations

export type Decl struct {
	Node;
	Exported bool;
	Ident *Expr;  // nil for ()-style declarations
	Typ *Type;
	Val *Expr;
	// list of *Decl for ()-style declarations
	// list of *Stat for func declarations (or nil for forward decl)
	List *array.Array; End int;
}


export func NewDecl(pos, tok int, exported bool) *Decl {
	d := new(Decl);
	d.Pos, d.Tok, d.Exported = pos, tok, exported;
	return d;
}


export var BadDecl = NewDecl(0, Scanner.ILLEGAL, false);


// ----------------------------------------------------------------------------
// Program

export type Comment struct {
	Pos int;
	Text string;
}


export func NewComment(pos int, text string) *Comment {
	c := new(Comment);
	c.Pos, c.Text = pos, text;
	return c;
}


export type Program struct {
	Pos int;  // tok is Scanner.PACKAGE
	Ident *Expr;
	Decls *array.Array;
	Comments *array.Array;
}


export func NewProgram(pos int) *Program {
	p := new(Program);
	p.Pos = pos;
	return p;
}
