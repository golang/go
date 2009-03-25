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
// sizes a bit. Then, embed Position field so we can get rid of
// most of the Pos() methods.


type (
	ExprVisitor interface;
	StmtVisitor interface;
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


// All statement nodes implement the Stmt interface.
type Stmt interface {
	// For a (dynamic) node type X, calling Visit with a statement
	// visitor v invokes the node-specific DoX function of the visitor.
	//
	Visit(v StmtVisitor);
	
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

// Support types.
type (
	Ident struct;
	StringLit struct;
	FunctionType struct;
	BlockStmt struct;

	// A Field represents a Field declaration list in a struct type,
	// a method in an interface type, or a parameter/result declaration
	// in a signature.
	Field struct {
		Doc Comments;  // associated documentation; or nil
		Names []*Ident;  // field/method/parameter names; nil if anonymous field
		Type Expr;  // field/method/parameter type
		Tag []*StringLit;  // field tag; nil if no tag
	};
)


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

	// An Ellipsis node stands for the "..." type in a
	// parameter list or the "..." length in an array type.
	//
	Ellipsis struct {
		Pos_ Position;  // position of "..."
	};

	// An IntLit node represents an integer literal.
	IntLit struct {
		Pos_ Position;  // literal string position
		Lit []byte;  // literal string; e.g. 42 or 0x7f
	};

	// A FloatLit node represents a floating-point literal.
	FloatLit struct {
		Pos_ Position;  // literal string position
		Lit []byte;  // literal string; e.g. 3.14 or 1e-9
	};

	// A CharLit node represents a character literal.
	CharLit struct {
		Pos_ Position;  // literal string position
		Lit []byte;  // literal string, including quotes; e.g. 'a' or '\x7f'
	};

	// A StringLit node represents a string literal.
	StringLit struct {
		Pos_ Position;  // literal string position
		Lit []byte;  // literal string, including quotes; e.g. "foo" or `\m\n\o`
	};

	// A StringList node represents a sequence of adjacent string literals.
	// A single string literal (common case) is represented by a StringLit
	// node; StringList nodes are used only if there are two or more string
	// literals in a sequence.
	//
	StringList struct {
		Strings []*StringLit;  // list of strings, len(Strings) > 1
	};

	// A FunctionLit node represents a function literal.
	FunctionLit struct {
		Type *FunctionType;  // function type
		Body *BlockStmt;  // function body
	};

	// A CompositeLit node represents a composite literal.
	CompositeLit struct {
		Type Expr;  // literal type
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

	// A SelectorExpr node represents an expression followed by a selector.
	SelectorExpr struct {
		X Expr;  // expression
		Sel *Ident;  // field selector
	};

	// An IndexExpr node represents an expression followed by an index.
	IndexExpr struct {
		X Expr;  // expression
		Index Expr;  // index expression
	};

	// A SliceExpr node represents an expression followed by a slice.
	SliceExpr struct {
		X Expr;  // expression
		Begin, End Expr;  // slice range
	};

	// A TypeAssertExpr node represents an expression followed by a
	// type assertion.
	//
	TypeAssertExpr struct {
		X Expr;  // expression
		Type Expr;  // asserted type
	};

	// A CallExpr node represents an expression followed by an argument list.
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
	// An ArrayType node represents an array type.
	ArrayType struct {
		Lbrack Position;  // position of "["
		Len Expr;  // possibly an Ellipsis node for [...]T array types
		Elt Expr;  // element type
	};

	// A SliceType node represents a slice type.
	SliceType struct {
		Lbrack Position;  // position of "["
		Elt Expr;  // element type
	};

	// A StructType node represents a struct type.
	StructType struct {
		Struct, Lbrace Position;  // positions of "struct" keyword, "{"
		Fields []*Field;  // list of field declarations; nil if forward declaration
		Rbrace Position;  // position of "}"
	};

	// Pointer types are represented via StarExpr nodes.

	// A FunctionType node represents a function type.
	FunctionType struct {
		Func Position;  // position of "func" keyword
		Params []*Field;  // (incoming) parameters
		Results []*Field;  // (outgoing) results
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
		Dir ChanDir;  // channel direction
		Value Expr;  // value type
	};
)


// Pos() implementations for all expression/type nodes.
//
func (x *BadExpr) Pos() Position  { return x.Pos_; }
func (x *Ident) Pos() Position  { return x.Pos_; }
func (x *IntLit) Pos() Position  { return x.Pos_; }
func (x *FloatLit) Pos() Position  { return x.Pos_; }
func (x *CharLit) Pos() Position  { return x.Pos_; }
func (x *StringLit) Pos() Position  { return x.Pos_; }
func (x *StringList) Pos() Position  { return x.Strings[0].Pos(); }
func (x *FunctionLit) Pos() Position  { return x.Type.Func; }
func (x *CompositeLit) Pos() Position  { return x.Type.Pos(); }
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
	DoIntLit(x *IntLit);
	DoFloatLit(x *FloatLit);
	DoCharLit(x *CharLit);
	DoStringLit(x *StringLit);
	DoStringList(x *StringList);
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
func (x *Ellipsis) Visit(v ExprVisitor) { v.DoEllipsis(x); }
func (x *IntLit) Visit(v ExprVisitor) { v.DoIntLit(x); }
func (x *FloatLit) Visit(v ExprVisitor) { v.DoFloatLit(x); }
func (x *CharLit) Visit(v ExprVisitor) { v.DoCharLit(x); }
func (x *StringLit) Visit(v ExprVisitor) { v.DoStringLit(x); }
func (x *StringList) Visit(v ExprVisitor) { v.DoStringList(x); }
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

func (x *ArrayType) Visit(v ExprVisitor) { v.DoArrayType(x); }
func (x *SliceType) Visit(v ExprVisitor) { v.DoSliceType(x); }
func (x *StructType) Visit(v ExprVisitor) { v.DoStructType(x); }
func (x *FunctionType) Visit(v ExprVisitor) { v.DoFunctionType(x); }
func (x *InterfaceType) Visit(v ExprVisitor) { v.DoInterfaceType(x); }
func (x *MapType) Visit(v ExprVisitor) { v.DoMapType(x); }
func (x *ChannelType) Visit(v ExprVisitor) { v.DoChannelType(x); }


// ----------------------------------------------------------------------------
// Statements

// A statement is represented by a tree consisting of one
// or more of the following concrete statement nodes.
//
type (
	// A BadStmt node is a placeholder for statements containing
	// syntax errors for which no correct statement nodes can be
	// created.
	//
	BadStmt struct {
		Pos_ Position;  // beginning position of bad statement
	};

	// A DeclStmt node represents a declaration in a statement list.
	DeclStmt struct {
		Decl Decl;
	};

	// An EmptyStmt node represents an empty statement.
	// The "position" of the empty statement is the position
	// of the immediately preceeding semicolon.
	//
	EmptyStmt struct {
		Semicolon Position;  // position of preceeding ";"
	};

	// A LabeledStmt node represents a labeled statement.
	LabeledStmt struct {
		Label *Ident;
		Stmt Stmt;
	};

	// An ExprStmt node represents a (stand-alone) expression
	// in a statement list.
	//
	ExprStmt struct {
		X Expr;  // expression
	};

	// An IncDecStmt node represents an increment or decrement statement.
	IncDecStmt struct {
		X Expr;
		Tok int;  // INC or DEC
	};

	// An AssignStmt node represents an assignment or
	// a short variable declaration.
	AssignStmt struct {
		Lhs []Expr;
		Pos_ Position;  // token position
		Tok int;  // assignment token, DEFINE
		Rhs []Expr;
	};

	// A GoStmt node represents a go statement.
	GoStmt struct {
		Go Position;  // position of "go" keyword
		Call *CallExpr;
	};

	// A DeferStmt node represents a defer statement.
	DeferStmt struct {
		Defer Position;  // position of "defer" keyword
		Call *CallExpr;
	};

	// A ReturnStmt node represents a return statement.
	ReturnStmt struct {
		Return Position;  // position of "return" keyword
		Results []Expr;
	};

	// A BranchStmt node represents a break, continue, goto,
	// or fallthrough statement.
	//
	BranchStmt struct {
		Pos_ Position;  // position of keyword
		Tok int;  // keyword token (BREAK, CONTINUE, GOTO, FALLTHROUGH)
		Label *Ident;
	};

	// A BlockStmt node represents a braced statement list.
	BlockStmt struct {
		Lbrace Position;
		List []Stmt;
		Rbrace Position;
	};

	// An IfStmt node represents an if statement.
	IfStmt struct {
		If Position;  // position of "if" keyword
		Init Stmt;
		Cond Expr;
		Body *BlockStmt;
		Else Stmt;
	};

	// A CaseClause represents a case of an expression switch statement.
	CaseClause struct {
		Case Position;  // position of "case" or "default" keyword
		Values []Expr;  // nil means default case
		Colon Position;  // position of ":"
		Body []Stmt;  // statement list; or nil
	};

	// A SwitchStmt node represents an expression switch statement.
	SwitchStmt struct {
		Switch Position;  // position of "switch" keyword
		Init Stmt;
		Tag Expr;
		Body *BlockStmt;  // CaseClauses only
	};

	// A TypeCaseClause represents a case of a type switch statement.
	TypeCaseClause struct {
		Case Position;  // position of "case" or "default" keyword
		Type Expr;  // nil means default case
		Colon Position;  // position of ":"
		Body []Stmt;  // statement list; or nil
	};

	// An TypeSwitchStmt node represents a type switch statement.
	TypeSwitchStmt struct {
		Switch Position;  // position of "switch" keyword
		Init Stmt;
		Assign Stmt;  // x := y.(type)
		Body *BlockStmt;  // TypeCaseClauses only
	};

	// A CommClause node represents a case of a select statement.
	CommClause struct {
		Case Position;  // position of "case" or "default" keyword
		Tok int;  // ASSIGN or DEFINE (valid only if Lhs != nil)
		Lhs, Rhs Expr;  // Rhs == nil means default case
		Colon Position;  // position of ":"
		Body []Stmt;  // statement list; or nil
	};

	// An SelectStmt node represents a select statement.
	SelectStmt struct {
		Select Position;  // position of "select" keyword
		Body *BlockStmt;  // CommClauses only
	};

	// A ForStmt represents a for statement.
	ForStmt struct {
		For Position;  // position of "for" keyword
		Init Stmt;
		Cond Expr;
		Post Stmt;
		Body *BlockStmt;
	};

	// A RangeStmt represents a for statement with a range clause.
	RangeStmt struct {
		For Position;  // position of "for" keyword
		Key, Value Expr;  // Value may be nil
		Pos_ Position;  // token position
		Tok int;  // ASSIGN or DEFINE
		X Expr;  // value to range over
		Body *BlockStmt;
	};
)


// Pos() implementations for all statement nodes.
//
func (s *BadStmt) Pos() Position { return s.Pos_; }
func (s *DeclStmt) Pos() Position { return s.Decl.Pos(); }
func (s *EmptyStmt) Pos() Position { return s.Semicolon; }
func (s *LabeledStmt) Pos() Position { return s.Label.Pos(); }
func (s *ExprStmt) Pos() Position { return s.X.Pos(); }
func (s *IncDecStmt) Pos() Position { return s.X.Pos(); }
func (s *AssignStmt) Pos() Position { return s.Lhs[0].Pos(); }
func (s *GoStmt) Pos() Position { return s.Go; }
func (s *DeferStmt) Pos() Position { return s.Defer; }
func (s *ReturnStmt) Pos() Position { return s.Return; }
func (s *BranchStmt) Pos() Position { return s.Pos_; }
func (s *BlockStmt) Pos() Position { return s.Lbrace; }
func (s *IfStmt) Pos() Position { return s.If; }
func (s *CaseClause) Pos() Position { return s.Case; }
func (s *SwitchStmt) Pos() Position { return s.Switch; }
func (s *TypeCaseClause) Pos() Position { return s.Case; }
func (s *TypeSwitchStmt) Pos() Position { return s.Switch; }
func (s *CommClause) Pos() Position { return s.Case; }
func (s *SelectStmt) Pos() Position { return s.Select; }
func (s *ForStmt) Pos() Position { return s.For; }
func (s *RangeStmt) Pos() Position { return s.For; }


// All statement nodes implement a Visit method which takes
// a StmtVisitor as argument. For a given node x of type X, and
// an implementation v of a StmtVisitor, calling x.Visit(v) will
// result in a call of v.DoX(x) (through a double-dispatch).
//
type StmtVisitor interface {
	DoBadStmt(s *BadStmt);
	DoDeclStmt(s *DeclStmt);
	DoEmptyStmt(s *EmptyStmt);
	DoLabeledStmt(s *LabeledStmt);
	DoExprStmt(s *ExprStmt);
	DoIncDecStmt(s *IncDecStmt);
	DoAssignStmt(s *AssignStmt);
	DoGoStmt(s *GoStmt);
	DoDeferStmt(s *DeferStmt);
	DoReturnStmt(s *ReturnStmt);
	DoBranchStmt(s *BranchStmt);
	DoBlockStmt(s *BlockStmt);
	DoIfStmt(s *IfStmt);
	DoCaseClause(s *CaseClause);
	DoSwitchStmt(s *SwitchStmt);
	DoTypeCaseClause(s *TypeCaseClause);
	DoTypeSwitchStmt(s *TypeSwitchStmt);
	DoCommClause(s *CommClause);
	DoSelectStmt(s *SelectStmt);
	DoForStmt(s *ForStmt);
	DoRangeStmt(s *RangeStmt);
}


// Visit() implementations for all statement nodes.
//
func (s *BadStmt) Visit(v StmtVisitor) { v.DoBadStmt(s); }
func (s *DeclStmt) Visit(v StmtVisitor) { v.DoDeclStmt(s); }
func (s *EmptyStmt) Visit(v StmtVisitor) { v.DoEmptyStmt(s); }
func (s *LabeledStmt) Visit(v StmtVisitor) { v.DoLabeledStmt(s); }
func (s *ExprStmt) Visit(v StmtVisitor) { v.DoExprStmt(s); }
func (s *IncDecStmt) Visit(v StmtVisitor) { v.DoIncDecStmt(s); }
func (s *AssignStmt) Visit(v StmtVisitor) { v.DoAssignStmt(s); }
func (s *GoStmt) Visit(v StmtVisitor) { v.DoGoStmt(s); }
func (s *DeferStmt) Visit(v StmtVisitor) { v.DoDeferStmt(s); }
func (s *ReturnStmt) Visit(v StmtVisitor) { v.DoReturnStmt(s); }
func (s *BranchStmt) Visit(v StmtVisitor) { v.DoBranchStmt(s); }
func (s *BlockStmt) Visit(v StmtVisitor) { v.DoBlockStmt(s); }
func (s *IfStmt) Visit(v StmtVisitor) { v.DoIfStmt(s); }
func (s *CaseClause) Visit(v StmtVisitor) { v.DoCaseClause(s); }
func (s *SwitchStmt) Visit(v StmtVisitor) { v.DoSwitchStmt(s); }
func (s *TypeCaseClause) Visit(v StmtVisitor) { v.DoTypeCaseClause(s); }
func (s *TypeSwitchStmt) Visit(v StmtVisitor) { v.DoTypeSwitchStmt(s); }
func (s *CommClause) Visit(v StmtVisitor) { v.DoCommClause(s); }
func (s *SelectStmt) Visit(v StmtVisitor) { v.DoSelectStmt(s); }
func (s *ForStmt) Visit(v StmtVisitor) { v.DoForStmt(s); }
func (s *RangeStmt) Visit(v StmtVisitor) { v.DoRangeStmt(s); }


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
		Doc Comments;  // associated documentation; or nil
		Import Position;  // position of "import" keyword
		Name *Ident;  // local package name or nil
		Path []*StringLit;  // package path
	};

	ConstDecl struct {
		Doc Comments;  // associated documentation; or nil
		Const Position;  // position of "const" keyword
		Names []*Ident;
		Type Expr;  // constant type or nil
		Values []Expr;
	};

	TypeDecl struct {
		Doc Comments;  // associated documentation; or nil
		Pos_ Position;  // position of "type" keyword
		Name *Ident;
		Type Expr;
	};

	VarDecl struct {
		Doc Comments;  // associated documentation; or nil
		Var Position;  // position of "var" keyword
		Names []*Ident;
		Type Expr;  // variable type or nil
		Values []Expr;
	};

	FuncDecl struct {
		Doc Comments;  // associated documentation; or nil
		Recv *Field;  // receiver (methods) or nil (functions)
		Name *Ident;  // function/method name
		Type *FunctionType;  // position of Func keyword, parameters and results
		Body *BlockStmt;  // function body or nil (forward declaration)
	};

	DeclList struct {
		Doc Comments;  // associated documentation; or nil
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
func (d *TypeDecl) Pos() Position { return d.Pos_; }
func (d *VarDecl) Pos() Position { return d.Var; }
func (d *FuncDecl) Pos() Position { return d.Type.Func; }
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
	Doc Comments;  // associated documentation; or nil
	Package Position;  // position of "package" keyword
	Name *Ident;  // package name
	Decls []Decl;  // top-level declarations
	Comments []*Comment;  // list of unassociated comments
}
