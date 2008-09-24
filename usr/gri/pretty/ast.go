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
	DoArrayType(x *ArrayType);
	DoStructType(x *StructType);
	DoMapType(x *MapType);
	DoChannelType(x *ChannelType);
	DoInterfaceType(x *InterfaceType);
	DoPointerType(x *PointerType);
	
	// Declarations
	DoConstDecl(x *ConstDecl);
	DoTypeDecl(x *TypeDecl);
	DoVarDecl(x *VarDecl);
	DoVarDeclList(x *VarDeclList);
	DoFuncDecl(x *FuncDecl);
	DoDeclaration(x *Declaration);
	
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
	DoCaseClause(x *CaseClause);
	DoSwitchStat(x *SwitchStat);
	DoReturnStat(x *ReturnStat);
	DoIncDecStat(x *IncDecStat);
	
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


export type Expr interface {
	Visit(x Visitor);
}


export type ArrayType struct {
	pos int;  // position of "["
	len_ Expr;
	elt Type;
}


export type StructType struct {
	pos int;  // position of "struct"
	fields *List;  // list of *VarDeclList
}


export type MapType struct {
	pos int;  // position of "map"
	key, val Type;
}


export type ChannelType struct {
	pos int;  // position of "chan" or "<-" (if before "chan")
	elt Type;
}


export type PointerType struct {
	pos int;  // position of "*"
	base Type;
}


export type InterfaceType struct {
}


export type FunctionType struct {
	pos int;  // position of "("
	recv *VarDeclList;
	params *List;  // list of *VarDeclList
	result *List;  // list of *VarDeclList
}


func (x *FunctionType)   Visit(v Visitor)  { v.DoFunctionType(x); }
func (x *ArrayType)      Visit(v Visitor)  { v.DoArrayType(x); }
func (x *StructType)     Visit(v Visitor)  { v.DoStructType(x); }
func (x *MapType)        Visit(v Visitor)  { v.DoMapType(x); }
func (x *ChannelType)    Visit(v Visitor)  { v.DoChannelType(x); }
func (x *PointerType)    Visit(v Visitor)  { v.DoPointerType(x); }
func (x *InterfaceType)  Visit(v Visitor)  { v.DoInterfaceType(x); }


// ----------------------------------------------------------------------------
// Declarations

export type Decl interface {
	Visit(x Visitor);
}


export type VarDeclList struct {
	idents *List;
	typ Type;
}


export type ConstDecl struct {
	ident *Ident;
	typ Type;
	val Expr;
}


export type TypeDecl struct {
	ident *Ident;
	typ Type;
}


export type VarDecl struct {
	idents *List;
	typ Type;
	vals *List;
}


export type Declaration struct {
	pos int;  // position of token
	tok int;
	decls *List;
}


export type FuncDecl struct {
	pos int;  // position of "func"
	ident *Ident;
	typ *FunctionType;
	body *Block;
}


func (x *VarDeclList)  Visit(v Visitor)  { v.DoVarDeclList(x); }
func (x *ConstDecl)    Visit(v Visitor)  { v.DoConstDecl(x); }
func (x *TypeDecl)     Visit(v Visitor)  { v.DoTypeDecl(x); }
func (x *VarDecl)      Visit(v Visitor)  { v.DoVarDecl(x); }
func (x *FuncDecl)     Visit(v Visitor)  { v.DoFuncDecl(x); }
func (x *Declaration)  Visit(v Visitor)  { v.DoDeclaration(x); }


// ----------------------------------------------------------------------------
// Expressions

export type Selector struct {
	pos int;  // position of "."
	x Expr;
	field string;
}


export type Index struct {
	pos int;  // position of "["
	x Expr;
	index Expr;
}


export type Call struct {
	pos int;  // position of "("
	fun Expr;
	args *List;
}


export type Pair struct {
	pos int;  // position of ":"
	x, y Expr;
}


export type Binary struct {
	pos int;  // position of operator tok
	tok int;
	x, y Expr;
}


export type Unary struct {
	pos int;  // position of operator tok
	tok int;
	x Expr;
}


export type Literal struct {
	pos int;  // position of literal
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
	pos int;  // position of "{"
	stats *List;
}


export type ExprStat struct {
	expr Expr;
}


export type Assignment struct {
	pos int;  // position of assignment token
	tok int;
	lhs, rhs *List;
}


export type IfStat struct {
	pos int;  // position of "if"
	init Stat;
	cond Expr;
	then, else_ *Block;
}


export type ForStat struct {
	pos int;  // position of "for"
	init Stat;
	cond Expr;
	post Stat;
	body *Block;
}


export type CaseClause struct {
	pos int;  // position of "case" or "default"
	exprs *List;  // nil if default case
	stats *List;  // list of Stat
	falls bool;
}


export type SwitchStat struct {
	pos int;  // position of "switch"
	init Stat;
	tag Expr;
	cases *List;  // list of *CaseClause
}


export type ReturnStat struct {
	pos int;  // position of "return"
	res *List;
}


export type IncDecStat struct {
	pos int;  // position of token
	tok int;
	expr Expr;
}


func (x *Block)       Visit(v Visitor)  { v.DoBlock(x); }
func (x *ExprStat)    Visit(v Visitor)  { v.DoExprStat(x); }
func (x *Assignment)  Visit(v Visitor)  { v.DoAssignment(x); }
func (x *IfStat)      Visit(v Visitor)  { v.DoIfStat(x); }
func (x *ForStat)     Visit(v Visitor)  { v.DoForStat(x); }
func (x *CaseClause)  Visit(v Visitor)  { v.DoCaseClause(x); }
func (x *SwitchStat)  Visit(v Visitor)  { v.DoSwitchStat(x); }
func (x *ReturnStat)  Visit(v Visitor)  { v.DoReturnStat(x); }
func (x *IncDecStat)  Visit(v Visitor)  { v.DoIncDecStat(x); }


// ----------------------------------------------------------------------------
// Program

export type Program struct {
	pos int;
	ident *Ident;
	decls *List;
}


func (x *Program) Visit(v Visitor)  { v.DoProgram(x); }
