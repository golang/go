// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package AST

import (
	"array";
	Scanner "scanner";
)


type (
	Any interface {};
	Type struct;
	Expr struct;
	Stat struct;
	Decl struct;
)


// ----------------------------------------------------------------------------
// All nodes have a source position and and token.

export type Node struct {
	pos, tok int;
}


// ----------------------------------------------------------------------------
// Expressions

export type Expr struct {
	Node;
	x, y *Expr;  // binary (x, y) and unary (y) expressions
	// TODO find a more space efficient way to hold these
	s string;  // identifiers and literals
	t *Type;  // type expressions, function literal types
	block *array.Array;  // stats for function literals
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


export func NewLit(pos, tok int, s string) *Expr {
	e := new(Expr);
	e.pos, e.tok, e.s = pos, tok, s;
	return e;
}


export var BadExpr = NewExpr(0, Scanner.ILLEGAL, nil, nil);


// ----------------------------------------------------------------------------
// Types

export const /* channel mode */ (
	FULL = iota;
	SEND;
	RECV;
)


export type Type struct {
	Node;
	expr *Expr;  // type name, array length
	mode int;  // channel mode
	key *Type;  // receiver type, map key
	elt *Type;  // array element, map or channel value, or pointer base type, result type
	list *array.Array;  // struct fields, interface methods, function parameters
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


export func NewType(pos, tok int) *Type {
	t := new(Type);
	t.pos, t.tok = pos, tok;
	return t;
}


// requires complete Type type
export func NewTypeExpr(t *Type) *Expr {
	e := new(Expr);
	e.pos, e.tok, e.t = t.pos, Scanner.TYPE, t;
	return e;
}


export var BadType = NewType(0, Scanner.ILLEGAL);


// ----------------------------------------------------------------------------
// Statements

export type Stat struct {
	Node;
	init, post *Stat;
	expr *Expr;
	block *array.Array;
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
	list *array.Array;
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
