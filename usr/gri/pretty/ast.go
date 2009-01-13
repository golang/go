// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package AST

import (
	"array";
	Scanner "scanner";
)


type (
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
	id int;  // unique id

	pos int;  // source position (< 0 if unknown position)
	kind int;  // object kind
	ident string;
	typ *Type;  // nil for packages
	pnolev int;  // >= 0: package no., <= 0: function nesting level, 0: global level
	
	// attached values
	block *array.Array; end int;  // stats for function literals; end of block pos
}



export var Universe_void_typ *Type  // initialized by Universe to Universe.void_typ
var ObjectId int;

export func NewObject(pos, kind int, ident string) *Object {
	obj := new(Object);
	obj.id = ObjectId;
	ObjectId++;
	
	obj.pos = pos;
	obj.kind = kind;
	obj.ident = ident;
	obj.typ = Universe_void_typ;
	obj.pnolev = 0;

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


func (scope *Scope) Add(obj* Object) {
	scope.entries[obj.ident] = obj;
}


func (scope *Scope) Insert(obj *Object) {
	if scope.LookupLocal(obj.ident) != nil {
		panic("obj already inserted");
	}
	scope.Add(obj);
}


func (scope *Scope) InsertImport(obj *Object) *Object {
	 p := scope.LookupLocal(obj.ident);
	 if p == nil {
		scope.Add(obj);
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
	pos int;  // source position (< 0 => unknown position)
	tok int;  // identifying token
}


// ----------------------------------------------------------------------------
// Expressions

export type Expr struct {
	Node;
	x, y *Expr;  // binary (x, y) and unary (y) expressions
	obj *Object;
}


func (x *Expr) Len() int {
	if x == nil {
		return 0;
	}
	n := 1;
	for ; x.tok == Scanner.COMMA; x = x.y {
		n++;
	}
	return n;
}


export func NewExpr(pos, tok int, x, y *Expr) *Expr {
	if x != nil && x.tok == Scanner.TYPE || y != nil && y.tok == Scanner.TYPE {
		panic("no type expression allowed");
	}
	e := new(Expr);
	e.pos, e.tok, e.x, e.y = pos, tok, x, y;
	return e;
}


// TODO probably don't need the tok parameter eventually
export func NewLit(tok int, obj *Object) *Expr {
	e := new(Expr);
	e.pos, e.tok, e.obj = obj.pos, tok, obj;
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
	id int;  // unique id

	ref int;  // for exporting only: >= 0 means already exported
	form int;  // type form
	size int;  // size in bytes
	obj *Object;  // primary type object or NULL
	scope *Scope;  // forwards, structs, interfaces, functions

	// syntactic components
	pos int;  // source position (< 0 if unknown position)
	expr *Expr;  // type name, array length
	mode int;  // channel mode
	key *Type;  // receiver type or map key
	elt *Type;  // array, map, channel or pointer element type, function result type
	list *array.Array; end int;  // struct fields, interface methods, function parameters
	scope *Scope;  // struct fields, methods
}


var TypeId int;

export func NewType(pos, form int) *Type {
	typ := new(Type);
	typ.id = TypeId;
	TypeId++;

	typ.ref = -1;  // not yet exported
	typ.pos = pos;
	typ.form = form;

	return typ;
}


func (t *Type) nfields() int {
	if t.list == nil {
		return 0;
	}
	nx, nt := 0, 0;
	for i, n := 0, t.list.Len(); i < n; i++ {
		if t.list.At(i).(*Expr).tok == Scanner.TYPE {
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


// requires complete Type.pos access
export func NewTypeExpr(typ *Type) *Expr {
	obj := NewObject(typ.pos, TYPE, "");
	obj.typ = typ;
	return NewLit(Scanner.TYPE, obj);
}


export var BadType = NewType(0, Scanner.ILLEGAL);


// ----------------------------------------------------------------------------
// Statements

export type Stat struct {
	Node;
	init, post *Stat;
	expr *Expr;
	block *array.Array; end int;  // block end position
	decl *Decl;
}


export func NewStat(pos, tok int) *Stat {
	s := new(Stat);
	s.pos, s.tok = pos, tok;
	return s;
}


export var BadStat = NewStat(0, Scanner.ILLEGAL);


// ----------------------------------------------------------------------------
// Declarations

export type Decl struct {
	Node;
	exported bool;
	ident *Expr;  // nil for ()-style declarations
	typ *Type;
	val *Expr;
	// list of *Decl for ()-style declarations
	// list of *Stat for func declarations (or nil for forward decl)
	list *array.Array; end int;
}


export func NewDecl(pos, tok int, exported bool) *Decl {
	d := new(Decl);
	d.pos, d.tok, d.exported = pos, tok, exported;
	return d;
}


export var BadDecl = NewDecl(0, Scanner.ILLEGAL, false);


// ----------------------------------------------------------------------------
// Program

export type Comment struct {
	pos int;
	text string;
}


export func NewComment(pos int, text string) *Comment {
	c := new(Comment);
	c.pos, c.text = pos, text;
	return c;
}


export type Program struct {
	pos int;  // tok is Scanner.PACKAGE
	ident *Expr;
	decls *array.Array;
	comments *array.Array;
}


export func NewProgram(pos int) *Program {
	p := new(Program);
	p.pos = pos;
	return p;
}
