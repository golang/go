// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The AST package declares the types used to represent
// syntax trees for Go source files.
//
package ast

import (
	"token";
	"unicode";
	"utf8";
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
// All nodes contain position information marking the beginning of
// the corresponding source text segment; it is accessible via the
// Pos accessor method. Nodes may contain additional position info
// for language constructs where comments may be found between parts
// of the construct (typically any larger, parenthesized subpart).
// That position information is needed to properly position comments
// when printing the construct.

// TODO: For comment positioning only the byte position and not
// a complete token.Position field is needed. May be able to trim
// node sizes a bit.


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
	Pos() token.Position;
}


// All statement nodes implement the Stmt interface.
type Stmt interface {
	// For a (dynamic) node type X, calling Visit with a statement
	// visitor v invokes the node-specific DoX function of the visitor.
	//
	Visit(v StmtVisitor);

	// Pos returns the (beginning) position of the statement.
	Pos() token.Position;
}


// All declaration nodes implement the Decl interface.
type Decl interface {
	// For a (dynamic) node type X, calling Visit with a declaration
	// visitor v invokes the node-specific DoX function of the visitor.
	//
	Visit(v DeclVisitor);

	// Pos returns the (beginning) position of the declaration.
	Pos() token.Position;
}


// ----------------------------------------------------------------------------
// Comments

// A Comment node represents a single //-style or /*-style comment.
type Comment struct {
	token.Position;  // beginning position of the comment
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
	FuncType struct;
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
		token.Position;  // beginning position of bad expression
	};

	// An Ident node represents an identifier.
	Ident struct {
		token.Position;  // identifier position
		Value string;  // identifier string (e.g. foobar)
	};

	// An Ellipsis node stands for the "..." type in a
	// parameter list or the "..." length in an array type.
	//
	Ellipsis struct {
		token.Position;  // position of "..."
	};

	// An IntLit node represents an integer literal.
	IntLit struct {
		token.Position;  // int literal position
		Value []byte;  // literal string; e.g. 42 or 0x7f
	};

	// A FloatLit node represents a floating-point literal.
	FloatLit struct {
		token.Position;  // float literal position
		Value []byte;  // literal string; e.g. 3.14 or 1e-9
	};

	// A CharLit node represents a character literal.
	CharLit struct {
		token.Position;  // char literal position
		Value []byte;  // literal string, including quotes; e.g. 'a' or '\x7f'
	};

	// A StringLit node represents a string literal.
	StringLit struct {
		token.Position;  // string literal position
		Value []byte;  // literal string, including quotes; e.g. "foo" or `\m\n\o`
	};

	// A StringList node represents a sequence of adjacent string literals.
	// A single string literal (common case) is represented by a StringLit
	// node; StringList nodes are used only if there are two or more string
	// literals in a sequence.
	//
	StringList struct {
		Strings []*StringLit;  // list of strings, len(Strings) > 1
	};

	// A FuncLit node represents a function literal.
	FuncLit struct {
		Type *FuncType;  // function type
		Body *BlockStmt;  // function body
	};

	// A CompositeLit node represents a composite literal.
	//
	CompositeLit struct {
		Type Expr;  // literal type
		Lbrace token.Position;  // position of "{"
		Elts []Expr;  // list of composite elements
		Rbrace token.Position;  // position of "}"
	};

	// A ParenExpr node represents a parenthesized expression.
	ParenExpr struct {
		token.Position;  // position of "("
		X Expr;  // parenthesized expression
		Rparen token.Position;  // position of ")"
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
		Lparen token.Position;  // position of "("
		Args []Expr;  // function arguments
		Rparen token.Position;  // positions of ")"
	};

	// A StarExpr node represents an expression of the form "*" Expression.
	// Semantically it could be a unary "*" expression, or a pointer type.
	StarExpr struct {
		token.Position;  // position of "*"
		X Expr;  // operand
	};

	// A UnaryExpr node represents a unary expression.
	// Unary "*" expressions are represented via StarExpr nodes.
	//
	UnaryExpr struct {
		token.Position;  // position of Op
		Op token.Token;  // operator
		X Expr;  // operand
	};

	// A BinaryExpr node represents a binary expression.
	//
	BinaryExpr struct {
		X Expr;  // left operand
		OpPos token.Position;  // position of Op
		Op token.Token;  // operator
		Y Expr;  // right operand
	};

	// A KeyValueExpr node represents (key : value) pairs
	// in composite literals.
	//
	KeyValueExpr struct {
		Key Expr;
		Colon token.Position;  // position of ":"
		Value Expr;
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
		token.Position;  // position of "["
		Len Expr;  // possibly an Ellipsis node for [...]T array types
		Elt Expr;  // element type
	};

	// A SliceType node represents a slice type.
	SliceType struct {
		token.Position;  // position of "["
		Elt Expr;  // element type
	};

	// A StructType node represents a struct type.
	StructType struct {
		token.Position;  // position of "struct" keyword
		Lbrace token.Position;  // position of "{"
		Fields []*Field;  // list of field declarations; nil if forward declaration
		Rbrace token.Position;  // position of "}"
	};

	// Pointer types are represented via StarExpr nodes.

	// A FuncType node represents a function type.
	FuncType struct {
		token.Position;  // position of "func" keyword
		Params []*Field;  // (incoming) parameters
		Results []*Field;  // (outgoing) results
	};

	// An InterfaceType node represents an interface type.
	InterfaceType struct {
		token.Position;  // position of "interface" keyword
		Lbrace token.Position;  // position of "{"
		Methods []*Field; // list of methods; nil if forward declaration
		Rbrace token.Position;  // position of "}"
	};

	// A MapType node represents a map type.
	MapType struct {
		token.Position;  // position of "map" keyword
		Key Expr;
		Value Expr;
	};

	// A ChanType node represents a channel type.
	ChanType struct {
		token.Position;  // position of "chan" keyword or "<-" (whichever comes first)
		Dir ChanDir;  // channel direction
		Value Expr;  // value type
	};
)


// Pos() implementations for expression/type where the position
// corresponds to the position of a sub-node.
//
func (x *StringList) Pos() token.Position  { return x.Strings[0].Pos(); }
func (x *FuncLit) Pos() token.Position  { return x.Type.Pos(); }
func (x *CompositeLit) Pos() token.Position  { return x.Type.Pos(); }
func (x *SelectorExpr) Pos() token.Position  { return x.X.Pos(); }
func (x *IndexExpr) Pos() token.Position  { return x.X.Pos(); }
func (x *SliceExpr) Pos() token.Position  { return x.X.Pos(); }
func (x *TypeAssertExpr) Pos() token.Position  { return x.X.Pos(); }
func (x *CallExpr) Pos() token.Position  { return x.Fun.Pos(); }
func (x *BinaryExpr) Pos() token.Position  { return x.X.Pos(); }
func (x *KeyValueExpr) Pos() token.Position  { return x.Key.Pos(); }


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
	DoFuncLit(x *FuncLit);
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
	DoKeyValueExpr(x *KeyValueExpr);

	// Type expressions
	DoEllipsis(x *Ellipsis);
	DoArrayType(x *ArrayType);
	DoSliceType(x *SliceType);
	DoStructType(x *StructType);
	DoFuncType(x *FuncType);
	DoInterfaceType(x *InterfaceType);
	DoMapType(x *MapType);
	DoChanType(x *ChanType);
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
func (x *FuncLit) Visit(v ExprVisitor) { v.DoFuncLit(x); }
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
func (x *KeyValueExpr) Visit(v ExprVisitor) { v.DoKeyValueExpr(x); }

func (x *ArrayType) Visit(v ExprVisitor) { v.DoArrayType(x); }
func (x *SliceType) Visit(v ExprVisitor) { v.DoSliceType(x); }
func (x *StructType) Visit(v ExprVisitor) { v.DoStructType(x); }
func (x *FuncType) Visit(v ExprVisitor) { v.DoFuncType(x); }
func (x *InterfaceType) Visit(v ExprVisitor) { v.DoInterfaceType(x); }
func (x *MapType) Visit(v ExprVisitor) { v.DoMapType(x); }
func (x *ChanType) Visit(v ExprVisitor) { v.DoChanType(x); }


// IsExported returns whether name is an exported Go symbol
// (i.e., whether it begins with an uppercase letter).
func IsExported(name string) bool {
	ch, len := utf8.DecodeRuneInString(name, 0);
	return unicode.IsUpper(ch);
}

// IsExported returns whether name is an exported Go symbol
// (i.e., whether it begins with an uppercase letter).
func (name *ast.Ident) IsExported() bool {
	return IsExported(name.Value);
}

func (name *ast.Ident) String() string {
	return name.Value;
}


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
		token.Position;  // beginning position of bad statement
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
		token.Position;  // position of preceeding ";"
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
		Tok token.Token;  // INC or DEC
	};

	// An AssignStmt node represents an assignment or
	// a short variable declaration.
	AssignStmt struct {
		Lhs []Expr;
		TokPos token.Position;  // position of Tok
		Tok token.Token;  // assignment token, DEFINE
		Rhs []Expr;
	};

	// A GoStmt node represents a go statement.
	GoStmt struct {
		token.Position;  // position of "go" keyword
		Call *CallExpr;
	};

	// A DeferStmt node represents a defer statement.
	DeferStmt struct {
		token.Position;  // position of "defer" keyword
		Call *CallExpr;
	};

	// A ReturnStmt node represents a return statement.
	ReturnStmt struct {
		token.Position;  // position of "return" keyword
		Results []Expr;
	};

	// A BranchStmt node represents a break, continue, goto,
	// or fallthrough statement.
	//
	BranchStmt struct {
		token.Position;  // position of Tok
		Tok token.Token;  // keyword token (BREAK, CONTINUE, GOTO, FALLTHROUGH)
		Label *Ident;
	};

	// A BlockStmt node represents a braced statement list.
	BlockStmt struct {
		token.Position;  // position of "{"
		List []Stmt;
		Rbrace token.Position;  // position of "}"
	};

	// An IfStmt node represents an if statement.
	IfStmt struct {
		token.Position;  // position of "if" keyword
		Init Stmt;
		Cond Expr;
		Body *BlockStmt;
		Else Stmt;
	};

	// A CaseClause represents a case of an expression switch statement.
	CaseClause struct {
		token.Position;  // position of "case" or "default" keyword
		Values []Expr;  // nil means default case
		Colon token.Position;  // position of ":"
		Body []Stmt;  // statement list; or nil
	};

	// A SwitchStmt node represents an expression switch statement.
	SwitchStmt struct {
		token.Position;  // position of "switch" keyword
		Init Stmt;
		Tag Expr;
		Body *BlockStmt;  // CaseClauses only
	};

	// A TypeCaseClause represents a case of a type switch statement.
	TypeCaseClause struct {
		token.Position;  // position of "case" or "default" keyword
		Type Expr;  // nil means default case
		Colon token.Position;  // position of ":"
		Body []Stmt;  // statement list; or nil
	};

	// An TypeSwitchStmt node represents a type switch statement.
	TypeSwitchStmt struct {
		token.Position;  // position of "switch" keyword
		Init Stmt;
		Assign Stmt;  // x := y.(type)
		Body *BlockStmt;  // TypeCaseClauses only
	};

	// A CommClause node represents a case of a select statement.
	CommClause struct {
		token.Position;  // position of "case" or "default" keyword
		Tok token.Token;  // ASSIGN or DEFINE (valid only if Lhs != nil)
		Lhs, Rhs Expr;  // Rhs == nil means default case
		Colon token.Position;  // position of ":"
		Body []Stmt;  // statement list; or nil
	};

	// An SelectStmt node represents a select statement.
	SelectStmt struct {
		token.Position;  // position of "select" keyword
		Body *BlockStmt;  // CommClauses only
	};

	// A ForStmt represents a for statement.
	ForStmt struct {
		token.Position;  // position of "for" keyword
		Init Stmt;
		Cond Expr;
		Post Stmt;
		Body *BlockStmt;
	};

	// A RangeStmt represents a for statement with a range clause.
	RangeStmt struct {
		token.Position;  // position of "for" keyword
		Key, Value Expr;  // Value may be nil
		TokPos token.Position;  // position of Tok
		Tok token.Token;  // ASSIGN, DEFINE
		X Expr;  // value to range over
		Body *BlockStmt;
	};
)


// Pos() implementations for statement nodes where the position
// corresponds to the position of a sub-node.
//
func (s *DeclStmt) Pos() token.Position { return s.Decl.Pos(); }
func (s *LabeledStmt) Pos() token.Position { return s.Label.Pos(); }
func (s *ExprStmt) Pos() token.Position { return s.X.Pos(); }
func (s *IncDecStmt) Pos() token.Position { return s.X.Pos(); }
func (s *AssignStmt) Pos() token.Position { return s.Lhs[0].Pos(); }


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

// A Spec node represents a single (non-parenthesized) import,
// constant, type, or variable declaration.
//
type (
	// The Spec type stands for any of *ImportSpec, *ValueSpec, and *TypeSpec.
	Spec interface {};

	// An ImportSpec node represents a single package import.
	ImportSpec struct {
		Doc Comments;  // associated documentation; or nil
		Name *Ident;  // local package name (including "."); or nil
		Path []*StringLit;  // package path
	};

	// A ValueSpec node represents a constant or variable declaration
	// (ConstSpec or VarSpec production).
	ValueSpec struct {
		Doc Comments;  // associated documentation; or nil
		Names []*Ident;
		Type Expr;  // value type; or nil
		Values []Expr;
	};

	// A TypeSpec node represents a type declaration (TypeSpec production).
	TypeSpec struct {
		Doc Comments;  // associated documentation; or nil
		Name *Ident;  // type name
		Type Expr;
	};
)


// A declaration is represented by one of the following declaration nodes.
//
type (
	// A BadDecl node is a placeholder for declarations containing
	// syntax errors for which no correct declaration nodes can be
	// created.
	//
	BadDecl struct {
		token.Position;  // beginning position of bad declaration
	};

	// A GenDecl node (generic declaration node) represents an import,
	// constant, type or variable declaration. A valid Lparen position
	// (Lparen.Line > 0) indicates a parenthesized declaration.
	//
	// Relationship between Tok value and Specs element type:
	//
	//	token.IMPORT  *ImportSpec
	//	token.CONST   *ValueSpec
	//	token.TYPE    *TypeSpec
	//	token.VAR     *ValueSpec
	//
	GenDecl struct {
		Doc Comments;  // associated documentation; or nil
		token.Position;  // position of Tok
		Tok token.Token;  // IMPORT, CONST, TYPE, VAR
		Lparen token.Position;  // position of '(', if any
		Specs []Spec;
		Rparen token.Position;  // position of ')', if any
	};

	// A FuncDecl node represents a function declaration.
	FuncDecl struct {
		Doc Comments;  // associated documentation; or nil
		Recv *Field;  // receiver (methods); or nil (functions)
		Name *Ident;  // function/method name
		Type *FuncType;  // position of Func keyword, parameters and results
		Body *BlockStmt;  // function body; or nil (forward declaration)
	};
)


// The position of a FuncDecl node is the position of its function type.
func (d *FuncDecl) Pos() token.Position  { return d.Type.Pos(); }


// All declaration nodes implement a Visit method which takes
// a DeclVisitor as argument. For a given node x of type X, and
// an implementation v of a DeclVisitor, calling x.Visit(v) will
// result in a call of v.DoX(x) (through a double-dispatch).
//
type DeclVisitor interface {
	DoBadDecl(d *BadDecl);
	DoGenDecl(d *GenDecl);
	DoFuncDecl(d *FuncDecl);
}


// Visit() implementations for all declaration nodes.
//
func (d *BadDecl) Visit(v DeclVisitor) { v.DoBadDecl(d); }
func (d *GenDecl) Visit(v DeclVisitor) { v.DoGenDecl(d); }
func (d *FuncDecl) Visit(v DeclVisitor) { v.DoFuncDecl(d); }


// ----------------------------------------------------------------------------
// Programs

// A Program node represents the root node of an AST
// for an entire source file.
//
type Program struct {
	Doc Comments;  // associated documentation; or nil
	token.Position;  // position of "package" keyword
	Name *Ident;  // package name
	Decls []Decl;  // top-level declarations
	Comments []*Comment;  // list of unassociated comments
}
