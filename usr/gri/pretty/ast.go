// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The AST package declares the types used to represent
// syntax trees for Go source files.
//
package ast

import (
	"token";
	"scanner";
)


// TODO rename Position to scanner.Position, possibly factor out
type Position scanner.Location


// TODO try to get rid of these
type (
	Block struct;
	Signature struct;
)


// ----------------------------------------------------------------------------
// Interfaces
//
// There are 3 main classes of nodes: Expressions and type nodes,
// statement nodes, and declaration nodes. The node names usually
// match the corresponding Go spec production names to which they
// correspond. The node fields correspond to the individual parts
// of the respective productions.
//
// Nodes contain selective position information: a position field
// marking the beginning of the corresponding source text segment
// if necessary; and specific position information for language
// constructs where comments may be found between parts of the
// construct (typically any larger, parenthesized subpart). The
// position information is needed to properly position comments
// when printing the construct.

// TODO: For comment positioning only the byte position and not
// a complete Position field is needed. May be able to trim node
// sizes a bit.


type (
	ExprVisitor interface;
	StatVisitor interface;
	DeclVisitor interface;
)


// All expression nodes implement the Expr interface.
type Expr interface {
	// For a (dynamic) node type X, calling Visit with an expression
	// visitor v invokes the node-specific DoX function of the visitor.
	//
	Visit(v ExprVisitor);
	
	// Pos returns the (beginning) position of the expression.
	Pos() Position;
}


// All statement nodes implement the Stat interface.
type Stat interface {
	// For a (dynamic) node type X, calling Visit with a statement
	// visitor v invokes the node-specific DoX function of the visitor.
	//
	Visit(v StatVisitor);
	
	// Pos returns the (beginning) position of the statement.
	Pos() Position;
}


// All declaration nodes implement the Decl interface.
type Decl interface {
	// For a (dynamic) node type X, calling Visit with a declaration
	// visitor v invokes the node-specific DoX function of the visitor.
	//
	Visit(v DeclVisitor);
	
	// Pos returns the (beginning) position of the declaration.
	Pos() Position;
}


// ----------------------------------------------------------------------------
// Comments

// A Comment node represents a single //-style or /*-style comment.
type Comment struct {
	Pos_ Position;  // beginning position of the comment
	Text []byte;  // the comment text (without '\n' for //-style comments)
	EndLine int;  // the line where the comment ends
}


// A Comments node represents a sequence of single comments
// with no other tokens and no empty lines between.
//
type Comments []*Comment


// ----------------------------------------------------------------------------
// Expressions and types

// An expression is represented by a tree consisting of one
// or more of the following concrete expression nodes.
//
type (
	// A BadExpr node is a placeholder for expressions containing
	// syntax errors for which no correct expression nodes can be
	// created.
	//
	BadExpr struct {
		Pos_ Position;  // beginning position of bad expression
	};

	// An Ident node represents an identifier.
	Ident struct {
		Pos_ Position;  // identifier position
		Lit []byte;  // identifier string (e.g. foobar)
	};

	// A BasicLit node represents a basic literal.
	BasicLit struct {
		Pos_ Position;  // literal string position
		Tok int;  // literal token (INT, FLOAT, CHAR, STRING)
		Lit []byte;  // literal string
	};

	// A StringLit node represents a sequence of string literals.
	StringLit struct {
		Strings []*BasicLit;  // sequence of strings
	};

	// A FunctionLit node represents a function literal.
	FunctionLit struct {
		Func Position;  // position of "func" keyword
		Typ *Signature;  // function signature
		Body *Block;  // function body
	};

	// A CompositeLit node represents a composite literal.
	CompositeLit struct {
		Typ Expr;  // literal type
		Lbrace Position;  // position of "{"
		Elts []Expr;  // list of composite elements
		Rbrace Position;  // position of "}"
	};

	// A ParenExpr node represents a parenthesized expression.
	ParenExpr struct {
		Lparen Position;  // position of "("
		X Expr;  // parenthesized expression
		Rparen Position;  // position of ")"
	};

	// A SelectorExpr node represents a primary expression followed by a selector.
	SelectorExpr struct {
		X Expr;  // primary expression
		Sel *Ident;  // field selector
	};

	// An IndexExpr node represents a primary expression followed by an index.
	IndexExpr struct {
		X Expr;  // primary expression
		Index Expr;  // index expression
	};

	// A SliceExpr node represents a primary expression followed by a slice.
	SliceExpr struct {
		X Expr;  // primary expression
		Begin, End Expr;  // slice range
	};

	// A TypeAssertExpr node represents a primary expression followed by a
	// type assertion.
	//
	TypeAssertExpr struct {
		X Expr;  // primary expression
		Typ Expr;  // asserted type
	};

	// A CallExpr node represents a primary expression followed by an argument list.
	CallExpr struct {
		Fun Expr;  // function expression
		Lparen Position;  // position of "("
		Args []Expr;  // function arguments
		Rparen Position;  // positions of ")"
	};

	// A StarExpr node represents an expression of the form "*" Expression.
	// Semantically it could be a unary "*" expression, or a pointer type.
	StarExpr struct {
		Star Position;  // position of "*"
		X Expr;  // operand
	};

	// A UnaryExpr node represents a unary expression.
	// Unary "*" expressions are represented via DerefExpr nodes.
	//
	UnaryExpr struct {
		Pos_ Position;  // token position
		Tok int;  // operator
		X Expr;  // operand
	};

	// A BinaryExpr node represents a binary expression.
	BinaryExpr struct {
		X Expr;  // left operand
		Pos_ Position;  // token position
		Tok int;  // operator
		Y Expr;  // right operand
	};
)


// The direction of a channel type is indicated by one
// of the following constants.
//
type ChanDir int
const (
	SEND ChanDir = 1 << iota;
	RECV;
)


// A type is represented by a tree consisting of one
// or more of the following type-specific expression
// nodes.
//
type (
	// An Ellipsis node stands for the "..." type in a
	// parameter list or the "..." length in an array type.
	//
	Ellipsis struct {  // neither a type nor an expression
		Pos_ Position;  // position of "..."
	};
	
	// An ArrayType node represents an array type.
	ArrayType struct {
		Lbrack Position;  // position of "["
		Len Expr;  // an Ellipsis node for [...]T array types
		Elt Expr;  // element type
	};

	// A SliceType node represents a slice type.
	SliceType struct {
		Lbrack Position;  // position of "["
		Elt Expr;  // element type
	};

	// A Field represents a Field declaration list in a struct type,
	// a method in an interface type, or a parameter declaration in
	// a signature.
	Field struct {
		Doc Comments;  // associated documentation (struct types only)
		Names []*Ident;  // field/method/parameter names; nil if anonymous field
		Typ Expr;  // field/method/parameter type
		Tag Expr;  // field tag; nil if no tag
	};

	// A StructType node represents a struct type.
	StructType struct {
		Struct, Lbrace Position;  // positions of "struct" keyword, "{"
		Fields []*Field;  // list of field declarations; nil if forward declaration
		Rbrace Position;  // position of "}"
	};

	// Note: pointer types are represented via StarExpr nodes.

	// A signature node represents the parameter and result
	// sections of a function type only.
	//
	Signature struct {
		Params []*Field;
		Result []*Field;
	};

	// A FunctionType node represents a function type.
	FunctionType struct {
		Func Position;  // position of "func" keyword
		Sig *Signature;
	};

	// An InterfaceType node represents an interface type.
	InterfaceType struct {
		Interface, Lbrace Position;  // positions of "interface" keyword, "{"
		Methods []*Field; // list of methods; nil if forward declaration
		Rbrace Position;  // position of "}"
	};

	// A MapType node represents a map type.
	MapType struct {
		Map Position;  // position of "map" keyword
		Key Expr;
		Value Expr;
	};

	// A ChannelType node represents a channel type.
	ChannelType struct {
		Pos_ Position;  // position of "chan" keyword or "<-" (whichever comes first)
		Dir ChanDir;
		Value Expr;  // value type
	};
)


// Pos() implementations for all expression/type nodes.
//
func (x *BadExpr) Pos() Position  { return x.Pos_; }
func (x *Ident) Pos() Position  { return x.Pos_; }
func (x *BasicLit) Pos() Position  { return x.Pos_; }
func (x *StringLit) Pos() Position  { return x.Strings[0].Pos(); }
func (x *FunctionLit) Pos() Position  { return x.Func; }
func (x *CompositeLit) Pos() Position  { return x.Typ.Pos(); }
func (x *ParenExpr) Pos() Position  { return x.Lparen; }
func (x *SelectorExpr) Pos() Position  { return x.X.Pos(); }
func (x *IndexExpr) Pos() Position  { return x.X.Pos(); }
func (x *SliceExpr) Pos() Position  { return x.X.Pos(); }
func (x *TypeAssertExpr) Pos() Position  { return x.X.Pos(); }
func (x *CallExpr) Pos() Position  { return x.Fun.Pos(); }
func (x *StarExpr) Pos() Position  { return x.Star; }
func (x *UnaryExpr) Pos() Position  { return x.Pos_; }
func (x *BinaryExpr) Pos() Position  { return x.X.Pos(); }

func (x *Ellipsis) Pos() Position { return x.Pos_; }
func (x *ArrayType) Pos() Position { return x.Lbrack; }
func (x *SliceType) Pos() Position { return x.Lbrack; }
func (x *StructType) Pos() Position { return x.Struct; }
func (x *FunctionType) Pos() Position { return x.Func; }
func (x *InterfaceType) Pos() Position { return x.Interface; }
func (x *MapType) Pos() Position { return x.Map; }
func (x *ChannelType) Pos() Position { return x.Pos_; }


// All expression/type nodes implement a Visit method which takes
// an ExprVisitor as argument. For a given node x of type X, and
// an implementation v of an ExprVisitor, calling x.Visit(v) will
// result in a call of v.DoX(x) (through a double-dispatch).
//
type ExprVisitor interface {
	// Expressions
	DoBadExpr(x *BadExpr);
	DoIdent(x *Ident);
	DoBasicLit(x *BasicLit);
	DoStringLit(x *StringLit);
	DoFunctionLit(x *FunctionLit);
	DoCompositeLit(x *CompositeLit);
	DoParenExpr(x *ParenExpr);
	DoSelectorExpr(x *SelectorExpr);
	DoIndexExpr(x *IndexExpr);
	DoSliceExpr(x *SliceExpr);
	DoTypeAssertExpr(x *TypeAssertExpr);
	DoCallExpr(x *CallExpr);
	DoStarExpr(x *StarExpr);
	DoUnaryExpr(x *UnaryExpr);
	DoBinaryExpr(x *BinaryExpr);

	// Type expressions
	DoEllipsis(x *Ellipsis);
	DoArrayType(x *ArrayType);
	DoSliceType(x *SliceType);
	DoStructType(x *StructType);
	DoFunctionType(x *FunctionType);
	DoInterfaceType(x *InterfaceType);
	DoMapType(x *MapType);
	DoChannelType(x *ChannelType);
}


// Visit() implementations for all expression/type nodes.
//
func (x *BadExpr) Visit(v ExprVisitor) { v.DoBadExpr(x); }
func (x *Ident) Visit(v ExprVisitor) { v.DoIdent(x); }
func (x *BasicLit) Visit(v ExprVisitor) { v.DoBasicLit(x); }
func (x *StringLit) Visit(v ExprVisitor) { v.DoStringLit(x); }
func (x *FunctionLit) Visit(v ExprVisitor) { v.DoFunctionLit(x); }
func (x *CompositeLit) Visit(v ExprVisitor) { v.DoCompositeLit(x); }
func (x *ParenExpr) Visit(v ExprVisitor) { v.DoParenExpr(x); }
func (x *SelectorExpr) Visit(v ExprVisitor) { v.DoSelectorExpr(x); }
func (x *IndexExpr) Visit(v ExprVisitor) { v.DoIndexExpr(x); }
func (x *SliceExpr) Visit(v ExprVisitor) { v.DoSliceExpr(x); }
func (x *TypeAssertExpr) Visit(v ExprVisitor) { v.DoTypeAssertExpr(x); }
func (x *CallExpr) Visit(v ExprVisitor) { v.DoCallExpr(x); }
func (x *StarExpr) Visit(v ExprVisitor) { v.DoStarExpr(x); }
func (x *UnaryExpr) Visit(v ExprVisitor) { v.DoUnaryExpr(x); }
func (x *BinaryExpr) Visit(v ExprVisitor) { v.DoBinaryExpr(x); }

func (x *Ellipsis) Visit(v ExprVisitor) { v.DoEllipsis(x); }
func (x *ArrayType) Visit(v ExprVisitor) { v.DoArrayType(x); }
func (x *SliceType) Visit(v ExprVisitor) { v.DoSliceType(x); }
func (x *StructType) Visit(v ExprVisitor) { v.DoStructType(x); }
func (x *FunctionType) Visit(v ExprVisitor) { v.DoFunctionType(x); }
func (x *InterfaceType) Visit(v ExprVisitor) { v.DoInterfaceType(x); }
func (x *MapType) Visit(v ExprVisitor) { v.DoMapType(x); }
func (x *ChannelType) Visit(v ExprVisitor) { v.DoChannelType(x); }


// ----------------------------------------------------------------------------
// Blocks

// A Block represents syntactic constructs of the form:
//
//   "{" StatementList "}"
//   ":" StatementList
//
type Block struct {
	Pos_ Position;
	Tok int;
	List []Stat;
	Rparen Position;  // position of closing "}" if present
}


// ----------------------------------------------------------------------------
// Statements

// A statement is represented by a tree consisting of one
// or more of the following concrete statement nodes.
//
type (
	// A BadStat node is a placeholder for statements containing
	// syntax errors for which no correct statement nodes can be
	// created.
	//
	BadStat struct {
		Pos_ Position;  // beginning position of bad statement
	};

	// A DeclStat node represents a declaration in a statement list.
	DeclStat struct {
		Decl Decl;
	};

	// An EmptyStat node represents an empty statement.
	// The "position" of the empty statement is the position
	// of the immediately preceeding semicolon.
	//
	EmptyStat struct {
		Semicolon Position;  // position of preceeding ";"
	};

	// A LabeledStat node represents a labeled statement.
	LabeledStat struct {
		Label *Ident;
		Stat Stat;
	};

	// An ExprStat node represents a (stand-alone) expression
	// in a statement list.
	//
	ExprStat struct {
		X Expr;  // expression
	};

	// An IncDecStat node represents an increment or decrement statement.
	IncDecStat struct {
		X Expr;
		Tok int;  // INC or DEC
	};

	// An AssignmentStat node represents an assignment or
	// a short variable declaration.
	AssignmentStat struct {
		Lhs []Expr;
		Pos_ Position;  // token position
		Tok int;  // assignment token, DEFINE
		Rhs []Expr;
	};

	// A GoStat node represents a go statement.
	GoStat struct {
		Go Position;  // position of "go" keyword
		Call Expr;
	};

	// A DeferStat node represents a defer statement.
	DeferStat struct {
		Defer Position;  // position of "defer" keyword
		Call Expr;
	};

	// A ReturnStat node represents a return statement.
	ReturnStat struct {
		Return Position;  // position of "return" keyword
		Results []Expr;
	};

	// A ControlFlowStat node represents a break, continue, goto,
	// or fallthrough statement.
	//
	ControlFlowStat struct {
		Pos_ Position;  // position of keyword
		Tok int;  // keyword token (BREAK, CONTINUE, GOTO, FALLTHROUGH)
		Label *Ident;
	};

	// A CompositeStat node represents a braced statement list.
	CompositeStat struct {
		Body *Block;
	};

	// An IfStat node represents an if statement.
	IfStat struct {
		If Position;  // position of "if" keyword
		Init Stat;
		Cond Expr;
		Body *Block;
		Else Stat;
	};

	// A CaseClause represents a case of an expression switch statement.
	CaseClause struct {
		Case Position;  // position of "case" or "default" keyword
		Values []Expr;  // nil means default case
		Body *Block;
	};

	// A SwitchStat node represents an expression switch statement.
	SwitchStat struct {
		Switch Position;  // position of "switch" keyword
		Init Stat;
		Tag Expr;
		Body *Block;  // CaseClauses only
	};

	// A TypeCaseClause represents a case of a type switch statement.
	TypeCaseClause struct {
		Case Position;  // position of "case" or "default" keyword
		Typ Expr;  // nil means default case
		Body *Block;
	};

	// An TypeSwitchStat node represents a type switch statement.
	TypeSwitchStat struct {
		Switch Position;  // position of "switch" keyword
		Init Stat;
		Assign Stat;  // x := y.(type)
		Body *Block;  // TypeCaseClauses only
	};

	// A CommClause node represents a case of a select statement.
	CommClause struct {
		Case Position;  // position of "case" or "default" keyword
		Tok int;  // ASSIGN, DEFINE (valid only if Lhs != nil)
		Lhs, Rhs Expr;  // Rhs == nil means default case
		Body *Block;
	};

	// An SelectStat node represents a select statement.
	SelectStat struct {
		Select Position;  // position of "select" keyword
		Body *Block;  // CommClauses only
	};

	// A ForStat represents a for statement.
	ForStat struct {
		For Position;  // position of "for" keyword
		Init Stat;
		Cond Expr;
		Post Stat;
		Body *Block;
	};

	// A RangeStat represents a for statement with a range clause.
	RangeStat struct {
		For Position;  // position of "for" keyword
		Range Stat;
		Body *Block;
	};
)


// Pos() implementations for all statement nodes.
//
func (s *BadStat) Pos() Position { return s.Pos_; }
func (s *DeclStat) Pos() Position { return s.Decl.Pos(); }
func (s *EmptyStat) Pos() Position { return s.Semicolon; }
func (s *LabeledStat) Pos() Position { return s.Label.Pos(); }
func (s *ExprStat) Pos() Position { return s.X.Pos(); }
func (s *IncDecStat) Pos() Position { return s.X.Pos(); }
func (s *AssignmentStat) Pos() Position { return s.Lhs[0].Pos(); }
func (s *GoStat) Pos() Position { return s.Go; }
func (s *DeferStat) Pos() Position { return s.Defer; }
func (s *ReturnStat) Pos() Position { return s.Return; }
func (s *ControlFlowStat) Pos() Position { return s.Pos_; }
func (s *CompositeStat) Pos() Position { return s.Body.Pos_; }
func (s *IfStat) Pos() Position { return s.If; }
func (s *CaseClause) Pos() Position { return s.Case; }
func (s *SwitchStat) Pos() Position { return s.Switch; }
func (s *TypeCaseClause) Pos() Position { return s.Case; }
func (s *TypeSwitchStat) Pos() Position { return s.Switch; }
func (s *CommClause) Pos() Position { return s.Case; }
func (s *SelectStat) Pos() Position { return s.Select; }
func (s *ForStat) Pos() Position { return s.For; }
func (s *RangeStat) Pos() Position { return s.For; }


// All statement nodes implement a Visit method which takes
// a StatVisitor as argument. For a given node x of type X, and
// an implementation v of a StatVisitor, calling x.Visit(v) will
// result in a call of v.DoX(x) (through a double-dispatch).
//
type StatVisitor interface {
	DoBadStat(s *BadStat);
	DoDeclStat(s *DeclStat);
	DoEmptyStat(s *EmptyStat);
	DoLabeledStat(s *LabeledStat);
	DoExprStat(s *ExprStat);
	DoIncDecStat(s *IncDecStat);
	DoAssignmentStat(s *AssignmentStat);
	DoGoStat(s *GoStat);
	DoDeferStat(s *DeferStat);
	DoReturnStat(s *ReturnStat);
	DoControlFlowStat(s *ControlFlowStat);
	DoCompositeStat(s *CompositeStat);
	DoIfStat(s *IfStat);
	DoCaseClause(s *CaseClause);
	DoSwitchStat(s *SwitchStat);
	DoTypeCaseClause(s *TypeCaseClause);
	DoTypeSwitchStat(s *TypeSwitchStat);
	DoCommClause(s *CommClause);
	DoSelectStat(s *SelectStat);
	DoForStat(s *ForStat);
	DoRangeStat(s *RangeStat);
}


// Visit() implementations for all statement nodes.
//
func (s *BadStat) Visit(v StatVisitor) { v.DoBadStat(s); }
func (s *DeclStat) Visit(v StatVisitor) { v.DoDeclStat(s); }
func (s *EmptyStat) Visit(v StatVisitor) { v.DoEmptyStat(s); }
func (s *LabeledStat) Visit(v StatVisitor) { v.DoLabeledStat(s); }
func (s *ExprStat) Visit(v StatVisitor) { v.DoExprStat(s); }
func (s *IncDecStat) Visit(v StatVisitor) { v.DoIncDecStat(s); }
func (s *AssignmentStat) Visit(v StatVisitor) { v.DoAssignmentStat(s); }
func (s *GoStat) Visit(v StatVisitor) { v.DoGoStat(s); }
func (s *DeferStat) Visit(v StatVisitor) { v.DoDeferStat(s); }
func (s *ReturnStat) Visit(v StatVisitor) { v.DoReturnStat(s); }
func (s *ControlFlowStat) Visit(v StatVisitor) { v.DoControlFlowStat(s); }
func (s *CompositeStat) Visit(v StatVisitor) { v.DoCompositeStat(s); }
func (s *IfStat) Visit(v StatVisitor) { v.DoIfStat(s); }
func (s *CaseClause) Visit(v StatVisitor) { v.DoCaseClause(s); }
func (s *SwitchStat) Visit(v StatVisitor) { v.DoSwitchStat(s); }
func (s *TypeCaseClause) Visit(v StatVisitor) { v.DoTypeCaseClause(s); }
func (s *TypeSwitchStat) Visit(v StatVisitor) { v.DoTypeSwitchStat(s); }
func (s *CommClause) Visit(v StatVisitor) { v.DoCommClause(s); }
func (s *SelectStat) Visit(v StatVisitor) { v.DoSelectStat(s); }
func (s *ForStat) Visit(v StatVisitor) { v.DoForStat(s); }
func (s *RangeStat) Visit(v StatVisitor) { v.DoRangeStat(s); }


// ----------------------------------------------------------------------------
// Declarations

// A declaration is represented by one of the following declaration nodes.
//
type (	
	// A BadDecl node is a placeholder for declarations containing
	// syntax errors for which no correct declaration nodes can be
	// created.
	//
	BadDecl struct {
		Pos_ Position;  // beginning position of bad declaration
	};

	ImportDecl struct {
		Doc Comments;  // associated documentation
		Import Position;  // position of "import" keyword
		Name *Ident;  // local package name or nil
		Path *StringLit;  // package path
	};

	ConstDecl struct {
		Doc Comments;  // associated documentation
		Const Position;  // position of "const" keyword
		Names []*Ident;
		Typ Expr;  // constant type or nil
		Values []Expr;
	};

	TypeDecl struct {
		Doc Comments;  // associated documentation
		Type Position;  // position of "type" keyword
		Name *Ident;
		Typ Expr;
	};

	VarDecl struct {
		Doc Comments;  // associated documentation
		Var Position;  // position of "var" keyword
		Names []*Ident;
		Typ Expr;  // variable type or nil
		Values []Expr;
	};

	FuncDecl struct {
		Doc Comments;  // associated documentation
		Func Position;  // position of "func" keyword
		Recv *Field;  // receiver (methods) or nil (functions)
		Name *Ident;  // function/method name
		Sig *Signature;  // parameters and results
		Body *Block;  // function body or nil (forward declaration)
	};

	DeclList struct {
		Doc Comments;  // associated documentation
		Pos_ Position;  // position of token
		Tok int;  // IMPORT, CONST, VAR, TYPE
		Lparen Position;  // position of '('
		List []Decl;  // the list of parenthesized declarations
		Rparen Position;  // position of ')'
	};
)


// Pos() implementations for all declaration nodes.
//
func (d *BadDecl) Pos() Position { return d.Pos_; }
func (d *ImportDecl) Pos() Position { return d.Import; }
func (d *ConstDecl) Pos() Position { return d.Const; }
func (d *TypeDecl) Pos() Position { return d.Type; }
func (d *VarDecl) Pos() Position { return d.Var; }
func (d *FuncDecl) Pos() Position { return d.Func; }
func (d *DeclList) Pos() Position { return d.Lparen; }


// All declaration nodes implement a Visit method which takes
// a DeclVisitor as argument. For a given node x of type X, and
// an implementation v of a DeclVisitor, calling x.Visit(v) will
// result in a call of v.DoX(x) (through a double-dispatch).
//
type DeclVisitor interface {
	DoBadDecl(d *BadDecl);
	DoImportDecl(d *ImportDecl);
	DoConstDecl(d *ConstDecl);
	DoTypeDecl(d *TypeDecl);
	DoVarDecl(d *VarDecl);
	DoFuncDecl(d *FuncDecl);
	DoDeclList(d *DeclList);
}


// Visit() implementations for all declaration nodes.
//
func (d *BadDecl) Visit(v DeclVisitor) { v.DoBadDecl(d); }
func (d *ImportDecl) Visit(v DeclVisitor) { v.DoImportDecl(d); }
func (d *ConstDecl) Visit(v DeclVisitor) { v.DoConstDecl(d); }
func (d *TypeDecl) Visit(v DeclVisitor) { v.DoTypeDecl(d); }
func (d *VarDecl) Visit(v DeclVisitor) { v.DoVarDecl(d); }
func (d *FuncDecl) Visit(v DeclVisitor) { v.DoFuncDecl(d); }
func (d *DeclList) Visit(v DeclVisitor) { v.DoDeclList(d); }


// ----------------------------------------------------------------------------
// Packages

// A Package node represents the root node of an AST.
type Package struct {
	Doc Comments;  // associated documentation
	Package Position;  // position of "package" keyword
	Name *Ident;  // package name
	Decls []Decl;  // top-level declarations
	Comments []*Comment;  // list of unassociated comments
}
