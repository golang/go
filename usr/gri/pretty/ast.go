// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package AST;

// ----------------------------------------------------------------------------
// Lists

export type Element interface {}


export type List struct {
	a *[] Element
}


func (p *List) len() int {
	return len(p.a);
}


func (p *List) at(i int) Element {
	return p.a[i];
}


func (p *List) Add (x Element) {
	a := p.a;
	n := len(a);

	if n == cap(a) {
		b := new([] interface {}, 2*n);
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
	p.a = new([] interface {}, 10);
	return p;
}


// ----------------------------------------------------------------------------
// Expressions

export type Expr interface {
}


export type Selector struct {
	pos int;
	x Expr;
	field string;
}


export type Index struct {
	pos int;
	x Expr;
	index Expr;
}


export type Pair struct {
	pos int;
	x, y Expr;
}


export type Binary struct {
	pos int;
	tok int;
	x, y Expr;
}


export type Unary struct {
	pos int;
	tok int;
	x Expr;
}


export type Literal struct {
	pos int;
	tok int;
	val string;
}


// ----------------------------------------------------------------------------
// Statements


// ----------------------------------------------------------------------------
// Visitor

export type Visitor interface {
  DoBinary(x *Binary);
  //DoUnary(x *Unary);
  //DoLiteral(x *Literal);
}


func (x *Binary)  Visit(v Visitor)  { v.DoBinary(x); }
//func (x *Unary)   Visit(v Visitor)  { v.DoUnary(x); }
//func (x *Literal) Visit(v Visitor)  { v.DoLiteral(x); }
