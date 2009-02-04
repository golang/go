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


type (
	Object struct;
	Type struct;

	Block struct;
	Expr interface;
	Stat struct;
	Decl struct;
)


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

	// attached values
	Body *Block;  // function body
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
// All nodes have a source position and a token.

type Node struct {
	Pos int;  // source position (< 0 => unknown position)
	Tok int;  // identifying token
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
		scope = scope.Parent;
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
	Expr Expr;  // type name, array length
	Mode int;  // channel mode
	Key *Type;  // receiver type or map key
	Elt *Type;  // type name type, array, map, channel or pointer element type, function result type
	List *array.Array; End int;  // struct fields, interface methods, function parameters
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


var BadType = NewType(0, Scanner.ILLEGAL);


// ----------------------------------------------------------------------------
// Blocks
//
// Syntactic constructs of the form:
//
//   "{" StatementList "}"
//   ":" StatementList

type Block struct {
	Node;
	List *array.Array;
	End int;  // position of closing "}" if present
}


func NewBlock(pos, tok int) *Block {
	assert(tok == Scanner.LBRACE || tok == Scanner.COLON);
	b := new(Block);
	b.Pos, b.Tok, b.List = pos, tok, array.New(0);
	return b;
}


// ----------------------------------------------------------------------------
// Expressions

type (
	Visitor interface;

	Expr interface {
		Pos() int;
		Visit(v Visitor);
	};

	BadExpr struct {
		Pos_ int;
	};

	Ident struct {
		Pos_ int;
		Obj *Object;
	};

	BinaryExpr struct {
		Pos_, Tok int;
		X, Y Expr;
	};

	UnaryExpr struct {
		Pos_, Tok int;
		X Expr;
	};

	BasicLit struct {
		Pos_, Tok int;
		Val string
	};

	FunctionLit struct {
		Pos_ int;  // position of "func"
		Typ *Type;
		Body *Block;
	};
	
	CompositeLit struct {
		Pos_ int;  // position of "{"
		Typ *Type;
		Elts Expr;
	};

	TypeLit struct {
		Typ *Type;
	};

	Selector struct {
		Pos_ int;  // position of "."
		X Expr;
		Sel *Ident;
	};

	TypeGuard struct {
		Pos_ int;  // position of "."
		X Expr;
		Typ *Type;
	};

	Index struct {
		Pos_ int;  // position of "["
		X, I Expr;
	};
	
	Call struct {
		Pos_ int;  // position of "("
		F, Args Expr
	};
)


type Visitor interface {
	DoBadExpr(x *BadExpr);
	DoIdent(x *Ident);
	DoBinaryExpr(x *BinaryExpr);
	DoUnaryExpr(x *UnaryExpr);
	DoBasicLit(x *BasicLit);
	DoFunctionLit(x *FunctionLit);
	DoCompositeLit(x *CompositeLit);
	DoTypeLit(x *TypeLit);
	DoSelector(x *Selector);
	DoTypeGuard(x *TypeGuard);
	DoIndex(x *Index);
	DoCall(x *Call);
}


func (x *BadExpr) Pos() int { return x.Pos_; }
func (x *Ident) Pos() int { return x.Pos_; }
func (x *BinaryExpr) Pos() int { return x.Pos_; }
func (x *UnaryExpr) Pos() int { return x.Pos_; }
func (x *BasicLit) Pos() int { return x.Pos_; }
func (x *FunctionLit) Pos() int { return x.Pos_; }
func (x *CompositeLit) Pos() int { return x.Pos_; }
func (x *TypeLit) Pos() int { return x.Typ.Pos; }
func (x *Selector) Pos() int { return x.Pos_; }
func (x *TypeGuard) Pos() int { return x.Pos_; }
func (x *Index) Pos() int { return x.Pos_; }
func (x *Call) Pos() int { return x.Pos_; }


func (x *BadExpr) Visit(v Visitor) { v.DoBadExpr(x); }
func (x *Ident) Visit(v Visitor) { v.DoIdent(x); }
func (x *BinaryExpr) Visit(v Visitor) { v.DoBinaryExpr(x); }
func (x *UnaryExpr) Visit(v Visitor) { v.DoUnaryExpr(x); }
func (x *BasicLit) Visit(v Visitor) { v.DoBasicLit(x); }
func (x *FunctionLit) Visit(v Visitor) { v.DoFunctionLit(x); }
func (x *CompositeLit) Visit(v Visitor) { v.DoCompositeLit(x); }
func (x *TypeLit) Visit(v Visitor) { v.DoTypeLit(x); }
func (x *Selector) Visit(v Visitor) { v.DoSelector(x); }
func (x *TypeGuard) Visit(v Visitor) { v.DoTypeGuard(x); }
func (x *Index) Visit(v Visitor) { v.DoIndex(x); }
func (x *Call) Visit(v Visitor) { v.DoCall(x); }



// Length of a comma-separated expression list.
func ExprLen(x Expr) int {
	if x == nil {
		return 0;
	}
	n := 1;
	for {
		if p, ok := x.(*BinaryExpr); ok && p.Tok == Scanner.COMMA {
			n++;
			x = p.Y;
		} else {
			break;
		}
	}
	return n;
}


func ExprAt(x Expr, i int) Expr {
	for j := 0; j < i; j++ {
		assert(x.(*BinaryExpr).Tok == Scanner.COMMA);
		x = x.(*BinaryExpr).Y;
	}
	if t, is_binary := x.(*BinaryExpr); is_binary && t.Tok == Scanner.COMMA {
		x = t.X;
	}
	return x;
}


func (t *Type) Nfields() int {
	if t.List == nil {
		return 0;
	}
	nx, nt := 0, 0;
	for i, n := 0, t.List.Len(); i < n; i++ {
		if dummy, ok := t.List.At(i).(*TypeLit); ok {
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


// ----------------------------------------------------------------------------
// Statements

type Stat struct {
	Node;
	Init, Post *Stat;
	Expr Expr;
	Body *Block;  // composite statement body
	Decl *Decl;  // declaration statement
}


func NewStat(pos, tok int) *Stat {
	s := new(Stat);
	s.Pos, s.Tok = pos, tok;
	return s;
}


var BadStat = NewStat(0, Scanner.ILLEGAL);


// ----------------------------------------------------------------------------
// Declarations

type Decl struct {
	Node;
	Ident Expr;  // nil for ()-style declarations
	Typ *Type;
	Val Expr;
	Body *Block;
	// list of *Decl for ()-style declarations
	List *array.Array; End int;
}


func NewDecl(pos, tok int) *Decl {
	d := new(Decl);
	d.Pos, d.Tok = pos, tok;
	return d;
}


var BadDecl = NewDecl(0, Scanner.ILLEGAL);


// ----------------------------------------------------------------------------
// Program

type Comment struct {
	Pos int;
	Text string;
}


func NewComment(pos int, text string) *Comment {
	c := new(Comment);
	c.Pos, c.Text = pos, text;
	return c;
}


type Program struct {
	Pos int;  // tok is Scanner.PACKAGE
	Ident Expr;
	Decls *array.Array;
	Comments *array.Array;
}


func NewProgram(pos int) *Program {
	p := new(Program);
	p.Pos = pos;
	return p;
}
