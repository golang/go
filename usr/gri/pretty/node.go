// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Node

import Scanner "scanner"

type Node interface {}

type (
	Type struct;
	Expr struct;
	Stat struct;
	Decl struct;
)


// ----------------------------------------------------------------------------
// Lists
//
// If p is a list and p == nil, then p.len() == 0.
// Thus, empty lists can be represented by nil.

export type List struct {
	a *[] Node;
}


func (p *List) len() int {
	if p == nil { return 0; }
	return len(p.a);
}


func (p *List) at(i int) Node {
	return p.a[i];
}


func (p *List) Add (x Node) {
	a := p.a;
	n := len(a);

	if n == cap(a) {
		b := new([] Node, 2*n);
		for i := 0; i < n; i++ {
			b[i] = a[i];
		}
		a = b;
	}

	a = a[0 : n + 1];
	a[n] = x;
	p.a = a;
}


/*
func (p *List) Print() {
	print("(");
	for i, n := 0, p.len(); i < n; i++ {
		if i > 0 {
			print(", ");
		}
		p.at(i).Print();
	}
	print(")");
}
*/


export func NewList() *List {
	p := new(List);
	p.a = new([] Node, 10) [0 : 0];
	return p;
}


// ----------------------------------------------------------------------------
// Types

export const /* channel mode */ (
	FULL = iota;
	SEND;
	RECV;
)


export type Type struct {
	pos, tok int;
	expr *Expr;  // type name, array length
	mode int;  // channel mode
	key *Type;  // map key
	elt *Type;  // array element, map or channel value, or pointer base type
	list *List;  // struct fields, interface methods, function parameters
}


export func NewType(pos, tok int) *Type {
	t := new(Type);
	t.pos, t.tok = pos, tok;
	return t;
}


// ----------------------------------------------------------------------------
// Expressions
//
// Expression pairs are represented as binary expressions with operator ":"
// Expression lists are represented as binary expressions with operator ","

export type Val struct {
	i int;
	f float;
	s string;
	t *Type;
}


export type Expr struct {
	pos, tok int;
	x, y *Expr;  // binary (x, y) and unary (y) expressions
	ident string;  // identifiers
	val *Val;  // literals
}


func (x *Expr) len() int {
	if x == nil {
		return 0;
	}
	n := 1;
	for ; x.tok == Scanner.COMMA; x = x.y {
		n++;
	}
	return n;
}


/*
func (x *Expr) Print() {
	switch {
	case x == nil:
		print("nil");
	case x.val != nil:
		print(x.val.s);
	default:
		if x.x == nil {
			print(Scanner.TokenName(x.tok));
		} else {
			x.x.Print();
			print(" ");
			print(Scanner.TokenName(x.tok));
			print(" ");
		}
		x.y.Print();
	}
}
*/


export func NewExpr(pos, tok int, x, y *Expr) *Expr {
	e := new(Expr);
	e.pos, e.tok, e.x, e.y = pos, tok, x, y;
	return e;
}


export func NewIdent(pos int, ident string) *Expr {
	e := new(Expr);
	e.pos, e.tok, e.ident = pos, Scanner.IDENT, ident;
	return e;
}


export func NewVal(pos, tok int, val *Val) *Expr {
	e := new(Expr);
	e.pos, e.tok, e.val = pos, tok, val;
	return e;
}


// ----------------------------------------------------------------------------
// Statements

export type Stat struct {
	pos, tok int;
	init *Stat;
	expr *Expr;
	post *Stat;
	block *List;
	decl *Decl;
}


export func NewStat(pos, tok int) *Stat {
	s := new(Stat);
	s.pos, s.tok = pos, tok;
	return s;
}


// ----------------------------------------------------------------------------
// Declarations

export type VarDeclList struct {
}


func (d *VarDeclList) Print() {
}


export type Decl struct {
	pos, tok int;
	exported bool;
	ident *Expr;  // nil for ()-style declarations
	typ *Type;
	val *Expr;
	// list of *Decl for ()-style declarations
	// list of *Stat for func declarations (or nil for forward decl)
	list *List;
}


export func NewDecl(pos, tok int, exported bool) *Decl {
	d := new(Decl);
	d.pos, d.tok, d.exported = pos, tok, exported;
	return d;
}


// ----------------------------------------------------------------------------
// Program

export type Program struct {
	pos int;  // tok is Scanner.PACKAGE
	ident *Expr;
	decls *List;
}


export func NewProgram(pos int) *Program {
	p := new(Program);
	p.pos = pos;
	return p;
}
