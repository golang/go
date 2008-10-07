// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package AST


// ----------------------------------------------------------------------------
// Visitor

type (
	Nil struct;
	Ident struct;
	ArrayType struct;
	StructType struct;
	MapType struct;
	ChannelType struct;
	PointerType struct;
	InterfaceType struct;
	FunctionType struct;
	VarDeclList struct;
	ImportDecl struct;
	ConstDecl struct;
	TypeDecl struct;
	VarDecl struct;
	Declaration struct;
	FuncDecl struct;
	MethodDecl struct;
	Selector struct;
	Index struct;
	Call struct;
	Pair struct;
	Binary struct;
	Unary struct;
	Literal struct;
	CompositeLit struct;
	FunctionLit struct;
	Label struct;
	Block struct;
	ExprStat struct;
	Assignment struct;
	ControlClause struct;
	IfStat struct;
	ForStat struct;
	CaseClause struct;
	SwitchStat struct;
	ReturnStat struct;
	IncDecStat struct;
	ControlFlowStat struct;
	GoStat struct;
	Program struct;
)

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
	DoImportDecl(x *ImportDecl);
	DoConstDecl(x *ConstDecl);
	DoTypeDecl(x *TypeDecl);
	DoVarDecl(x *VarDecl);
	DoVarDeclList(x *VarDeclList);
	DoFuncDecl(x *FuncDecl);
	DoMethodDecl(x *MethodDecl);
	DoDeclaration(x *Declaration);

	// Expressions
	DoBinary(x *Binary);
	DoUnary(x *Unary);
	DoLiteral(x *Literal);
	DoPair(x *Pair);
	DoIndex(x *Index);
	DoCall(x *Call);
	DoSelector(x *Selector);
	DoCompositeLit(x *CompositeLit);
	DoFunctionLit(x *FunctionLit);

	// Statements
	DoLabel(x *Label);
	DoBlock(x *Block);
	DoExprStat(x *ExprStat);
	DoAssignment(x *Assignment);
	DoIfStat(x *IfStat);
	DoForStat(x *ForStat);
	DoCaseClause(x *CaseClause);
	DoSwitchStat(x *SwitchStat);
	DoReturnStat(x *ReturnStat);
	DoIncDecStat(x *IncDecStat);
	DoControlFlowStat(x *ControlFlowStat);
	DoGoStat(x *GoStat);

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


export const /* chan mode */ (
	FULL = iota;
	RECV;
	SEND;
)

export type ChannelType struct {
	pos int;  // position of "chan" or "<-" (if before "chan")
	elt Type;
	mode int;
}


export type PointerType struct {
	pos int;  // position of "*"
	base Type;
}


export type InterfaceType struct {
	pos int;  // position of "interface"
	methods *List;  // list of *MethodDecl
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
	idents *List;  // possibly nil
	typ Type;
}


export type ImportDecl struct {
	ident *Ident;
	file string;
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


export type MethodDecl struct {
	ident *Ident;
	typ *FunctionType;
}


func (x *VarDeclList)  Visit(v Visitor)  { v.DoVarDeclList(x); }
func (x *ImportDecl)   Visit(v Visitor)  { v.DoImportDecl(x); }
func (x *ConstDecl)    Visit(v Visitor)  { v.DoConstDecl(x); }
func (x *TypeDecl)     Visit(v Visitor)  { v.DoTypeDecl(x); }
func (x *VarDecl)      Visit(v Visitor)  { v.DoVarDecl(x); }
func (x *FuncDecl)     Visit(v Visitor)  { v.DoFuncDecl(x); }
func (x *MethodDecl)   Visit(v Visitor)  { v.DoMethodDecl(x); }
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


export type CompositeLit struct {
	pos int;  // position of "{"
	typ Type;
	vals *List  // list of Expr
}


export type FunctionLit struct {
	pos int;  // position of "func"
	typ *FunctionType;
	body *Block;
}


func (x *Binary)       Visit(v Visitor)  { v.DoBinary(x); }
func (x *Unary)        Visit(v Visitor)  { v.DoUnary(x); }
func (x *Literal)      Visit(v Visitor)  { v.DoLiteral(x); }
func (x *Pair)         Visit(v Visitor)  { v.DoPair(x); }
func (x *Index)        Visit(v Visitor)  { v.DoIndex(x); }
func (x *Call)         Visit(v Visitor)  { v.DoCall(x); }
func (x *Selector)     Visit(v Visitor)  { v.DoSelector(x); }
func (x *CompositeLit) Visit(v Visitor)  { v.DoCompositeLit(x); }
func (x *FunctionLit)  Visit(v Visitor)  { v.DoFunctionLit(x); }


// ----------------------------------------------------------------------------
// Statements

export type Stat interface {
	Visit(x Visitor);
}


export type Label struct {
	pos int;  // position of ":"
	ident Expr;  // should be ident
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


export type ControlClause struct {
	init Stat;
	expr Expr;
	post Stat;
	has_init, has_expr, has_post bool;
}


export type IfStat struct {
	pos int;  // position of "if"
	ctrl *ControlClause;
	then *Block;
	else_ Stat;
	has_else bool;
}


export type ForStat struct {
	pos int;  // position of "for"
	ctrl *ControlClause;
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
	ctrl *ControlClause;
	cases *List;  // list of *CaseClause
}


export type ReturnStat struct {
	pos int;  // position of "return"
	res *List;  // list of Expr
}


export type IncDecStat struct {
	pos int;  // position of token
	tok int;
	expr Expr;
}


export type ControlFlowStat struct {
	pos int;  // position of token
	tok int;
	label *Ident;  // nil, if no label
}


export type GoStat struct {
	pos int;  // position of "go"
	expr Expr;
}


func (x *Block)            Visit(v Visitor)  { v.DoBlock(x); }
func (x *Label)            Visit(v Visitor)  { v.DoLabel(x); }
func (x *ExprStat)         Visit(v Visitor)  { v.DoExprStat(x); }
func (x *Assignment)       Visit(v Visitor)  { v.DoAssignment(x); }
func (x *IfStat)           Visit(v Visitor)  { v.DoIfStat(x); }
func (x *ForStat)          Visit(v Visitor)  { v.DoForStat(x); }
func (x *CaseClause)       Visit(v Visitor)  { v.DoCaseClause(x); }
func (x *SwitchStat)       Visit(v Visitor)  { v.DoSwitchStat(x); }
func (x *ReturnStat)       Visit(v Visitor)  { v.DoReturnStat(x); }
func (x *IncDecStat)       Visit(v Visitor)  { v.DoIncDecStat(x); }
func (x *ControlFlowStat)  Visit(v Visitor)  { v.DoControlFlowStat(x); }
func (x *GoStat)           Visit(v Visitor)  { v.DoGoStat(x); }


// ----------------------------------------------------------------------------
// Program

export type Program struct {
	pos int;
	ident *Ident;
	decls *List;
}


func (x *Program) Visit(v Visitor)  { v.DoProgram(x); }
