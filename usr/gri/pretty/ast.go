// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package AST


// ----------------------------------------------------------------------------
// Visitor

export type Visitor interface {
	// Basics
	DoNil(x *Nil);
	DoIdent(x *Ident);
	
	// Types
	DoFunctionType(x *FunctionType);
	
	// Declarations
	//DoVarDeclList(x *VarDeclList);
	DoFuncDecl(x *FuncDecl);
	
	// Expressions
	DoBinary(x *Binary);
	DoUnary(x *Unary);
	DoLiteral(x *Literal);
	DoPair(x *Pair);
	DoIndex(x *Index);
	DoCall(x *Call);
	DoSelector(x *Selector);
	
	// Statements
	DoBlock(x *Block);
	DoExprStat(x *ExprStat);
	DoAssignment(x *Assignment);
	DoIfStat(x *IfStat);
	DoForStat(x *ForStat);
	DoSwitch(x *Switch);
	DoReturn(x *Return);
	
	// Program
	DoProgram(x *Program);
}


// ----------------------------------------------------------------------------
// An AST Node

export type Node interface {
	Visit(x Visitor);
}


// ----------------------------------------------------------------------------
// Lists

export type List struct {
	a *[] Node
}


func (p *List) len() int {
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


export func NewList() *List {
	p := new(List);
	p.a = new([] Node, 10) [0 : 0];
	return p;
}


// ----------------------------------------------------------------------------
// Basics

export type Nil struct {
	// The Node "nil" value
}

export var NIL *Nil = new(Nil);


export type Ident struct {
	pos int;
	val string;
}


func (x *Nil)   Visit(v Visitor)  { v.DoNil(x); }
func (x *Ident) Visit(v Visitor)  { v.DoIdent(x); }


// ----------------------------------------------------------------------------
// Types

export type Type interface {
	Visit(x Visitor);
}


export type FunctionType struct {
	recv *VarDeclList;
	params *List;
	result *List;
}


func (x *FunctionType) Visit(v Visitor)  { v.DoFunctionType(x); }


// ----------------------------------------------------------------------------
// Declarations

export type Decl interface {
	Visit(x Visitor);
}


export type VarDeclList struct {
	idents *List;
	typ *Node;
}


export type FuncDecl struct {
	pos int;
	ident *Ident;
	typ *FunctionType;
	body *Block;
}


func (x *VarDeclList) Visit(v Visitor)  { /*v.DoVarDeclList(x);*/ }
func (x *FuncDecl)    Visit(v Visitor)  { v.DoFuncDecl(x); }


// ----------------------------------------------------------------------------
// Expressions

export type Expr interface {
	Visit(x Visitor);
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


export type Call struct {
	pos int;
	fun Expr;
	args *List;
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


func (x *Binary)   Visit(v Visitor)  { v.DoBinary(x); }
func (x *Unary)    Visit(v Visitor)  { v.DoUnary(x); }
func (x *Literal)  Visit(v Visitor)  { v.DoLiteral(x); }
func (x *Pair)     Visit(v Visitor)  { v.DoPair(x); }
func (x *Index)    Visit(v Visitor)  { v.DoIndex(x); }
func (x *Call)     Visit(v Visitor)  { v.DoCall(x); }
func (x *Selector) Visit(v Visitor)  { v.DoSelector(x); }


// ----------------------------------------------------------------------------
// Statements

export type Stat interface {
	Visit(x Visitor);
}


export type Block struct {
	pos int;
	stats *List;
}


export type ExprStat struct {
	expr Expr;
}


export type Assignment struct {
	pos int;
	tok int;
	lhs, rhs *List;
}


export type IfStat struct {
	pos int;
	init Stat;
	cond Expr;
	then, else_ *Block;
}


export type ForStat struct {
	pos int;
	body *Block;
}


export type Switch struct {
}


export type Return struct {
	pos int;
	res *List;
}


func (x *Block)       Visit(v Visitor)  { v.DoBlock(x); }
func (x *ExprStat)    Visit(v Visitor)  { v.DoExprStat(x); }
func (x *Assignment)  Visit(v Visitor)  { v.DoAssignment(x); }
func (x *IfStat)      Visit(v Visitor)  { v.DoIfStat(x); }
func (x *ForStat)     Visit(v Visitor)  { v.DoForStat(x); }
func (x *Switch)      Visit(v Visitor)  { v.DoSwitch(x); }
func (x *Return)      Visit(v Visitor)  { v.DoReturn(x); }


// ----------------------------------------------------------------------------
// Program

export type Program struct {
	pos int;
	ident *Ident;
	decls *List;
}


func (x *Program) Visit(v Visitor)  { v.DoProgram(x); }
