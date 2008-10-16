// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Node

import Scanner "scanner"


type (
	Node interface {};
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


func (p *List) set(i int, x Node) {
	p.a[i] = x;
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


export func NewList() *List {
	p := new(List);
	p.a = new([] Node, 10) [0 : 0];
	return p;
}


// ----------------------------------------------------------------------------
// Expressions

export type Expr struct {
	pos, tok int;
	x, y *Expr;  // binary (x, y) and unary (y) expressions
	s string;  // identifiers and literals
	t *Type;  // operands that are types
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


export func NewExpr(pos, tok int, x, y *Expr) *Expr {
	e := new(Expr);
	e.pos, e.tok, e.x, e.y = pos, tok, x, y;
	return e;
}


export func NewLit(pos, tok int, s string) *Expr {
	e := new(Expr);
	e.pos, e.tok, e.s = pos, tok, s;
	return e;
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
	key *Type;  // receiver type, map key
	elt *Type;  // array element, map or channel value, or pointer base type, result type
	list *List;  // struct fields, interface methods, function parameters
}


func (t *Type) nfields() int {
	nx, nt := 0, 0;
	for i, n := 0, t.list.len(); i < n; i++ {
		if t.list.at(i).(*Expr).tok == Scanner.TYPE {
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


// ----------------------------------------------------------------------------
// Statements

export type Stat struct {
	pos, tok int;
	init, post *Stat;
	lhs, expr *Expr;
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
