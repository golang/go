// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file defines a collection of interfaces providing an AST abstraction,
// for use in a type checker that operates on multiple concrete ASTs.

package types

import (
	"go/ast"
	"go/token"
)

type Pos interface {
	IsKnown() bool
	Before(pos Pos) bool
	String() string
}

// Nodes
type (
	Node interface {
		Pos() Pos
	}

	File interface {
		Package() Pos
		Name() Ident

		// TODO: devise a new schema for *Len methods. DeclsLen is too clunky.

		DeclsLen() int
		Decl(i int) Decl

		Node
	}

	FieldList interface {
		Len() int
		Field(i int) Field

		Node
	}

	Field interface {
		NamesLen() int
		Name(i int) Ident
		Type() Expr
		Tag() BasicLit

		Node
	}
)

// Decls
type (
	Decl interface {
		ADecl()
		Node
	}

	BadDecl interface {
		ABadDecl()
		Decl
	}

	GenDecl interface {
		Tok() token.Token
		SpecsLen() int
		Spec(i int) Spec

		AGenDecl()
		Decl
	}

	FuncDecl interface {
		Recv() FieldList
		Name() Ident
		Type() FuncType
		Body() BlockStmt

		AFuncDecl()
		Decl
	}
)

// Specs
type (
	Spec interface {
		ASpec()
		Node
	}

	// TODO: consolidate these specs with decls.

	ValueSpec interface {
		NamesLen() int
		Name(i int) Ident
		Type() Expr
		ValuesLen() int
		Value(i int) Expr

		AValueSpec()
		Spec
	}

	TypeSpec interface {
		Name() Ident
		Assign() Pos
		Type() Expr

		ATypeSpec()
		Spec
	}

	ImportSpec interface {
		Path() BasicLit
		Name() Ident

		AnImportSpec()
		Spec
	}
)

// Exprs
type (
	Expr interface {
		AnExpr()
		Node
	}

	Ident interface {
		Name() string

		AnIdent()
		Expr
	}

	SelectorExpr interface {
		X() Expr
		Sel() Ident

		ASelectorExpr()
		Expr
	}

	BadExpr interface {
		ABadExpr()
		Expr
	}

	DotDotDot interface {
		Elt() Expr

		ADotDotDot()
		Expr
	}

	BasicLit interface {
		Kind() token.Token
		Value() string

		ABasicLit()
		Expr
	}

	FuncLit interface {
		Type() Expr
		Body() BlockStmt

		AFuncLit()
		Expr
	}

	CompositeLit interface {
		Type() Expr
		EltsLen() int
		Elt(i int) Expr
		Rbrace() Pos

		ACompositeLit()
		Expr
	}

	ParenExpr interface {
		X() Expr

		AParenExpr()
		Expr
	}

	IndexExpr interface {
		X() Expr
		Index() Expr

		AnIndexExpr()
		Expr
	}

	SliceExpr interface {
		X() Expr
		Low() Expr
		High() Expr
		Max() Expr
		Slice3() bool
		Rbrack() Pos

		ASliceExpr()
		Expr
	}

	TypeAssertExpr interface {
		X() Expr
		Type() Expr

		ATypeAssertExpr()
		Expr
	}

	CallExpr interface {
		ArgsLen() int
		Arg(i int) Expr
		Fun() Expr
		Ellipsis() Pos
		Rparen() Pos

		ACallExpr()
		Expr
	}

	StarExpr interface {
		Expr
		AStarExpr()

		X() Expr
	}

	UnaryExpr interface {
		Expr
		AUnaryExpr()

		Op() token.Token
		X() Expr
	}

	BinaryExpr interface {
		Op() token.Token
		X() Expr
		Y() Expr

		ABinaryExpr()
		Expr
	}

	KeyValueExpr interface {
		Key() Expr
		Value() Expr

		AKeyValueExpr()
		Expr
	}

	ArrayType interface {
		Len() Expr
		Elt() Expr

		AnArrayType()
		Expr
	}

	StructType interface {
		Fields() FieldList

		AStructType()
		Expr
	}

	FuncType interface {
		Params() FieldList
		Results() FieldList

		AFuncType()
		Expr
	}

	InterfaceType interface {
		Methods() FieldList

		AnInterfaceType()
		Expr
	}

	MapType interface {
		Key() Expr
		Value() Expr

		AMapType()
		Expr
	}

	ChanType interface {
		// TODO: replace this return type
		Dir() ast.ChanDir
		Value() Expr

		AChanType()
		Expr
	}
)

type ExprList interface {
	Len() int
	Expr(i int) Expr
}

// Stmts
type (
	Stmt interface {
		AStmt()
		Node
	}

	BadStmt interface {
		ABadStmt()
		Stmt
	}

	DeclStmt interface {
		Decl() Decl

		ADeclStmt()
		Stmt
	}

	EmptyStmt interface {
		AnEmptyStmt()
		Stmt
	}

	LabeledStmt interface {
		Label() Ident
		Stmt() Stmt

		ALabeledStmt()
		Stmt
	}

	ExprStmt interface {
		X() Expr

		AnExprStmt()
		Stmt
	}

	SendStmt interface {
		Chan() Expr
		Value() Expr
		Arrow() Pos

		ASendStmt()
		Stmt
	}

	IncDecStmt interface {
		Tok() token.Token
		TokPos() Pos
		X() Expr

		AnIncDecStmt()
		Stmt
	}

	AssignStmt interface {
		Lhs() ExprList
		LhsLen() int
		LhsExpr(i int) Expr
		RhsLen() int
		RhsExpr(i int) Expr
		Rhs() ExprList
		Tok() token.Token
		TokPos() Pos

		AnAssignStmt()
		Stmt
	}

	GoStmt interface {
		Call() CallExpr

		AGoStmt()
		Stmt
	}

	DeferStmt interface {
		Call() CallExpr

		ADeferStmt()
		Stmt
	}

	ReturnStmt interface {
		Results() ExprList
		ResultsLen() int
		Result(i int) Expr
		Return() Pos

		AReturnStmt()
		Stmt
	}

	BranchStmt interface {
		Tok() token.Token
		Label() Ident

		ABranchStmt()
		Stmt
	}

	BlockStmt interface {
		List() StmtList
		Lbrace() Pos
		Rbrace() Pos

		ABlockStmt()
		Stmt
	}

	IfStmt interface {
		Init() Stmt
		Cond() Expr
		Body() BlockStmt
		Else() Stmt

		AnIfStmt()
		Stmt
	}

	CaseClause interface {
		ListLen() int
		Item(i int) Expr
		Body() StmtList

		ACaseClause()
		Stmt
	}

	SwitchStmt interface {
		Init() Stmt
		Tag() Expr
		Body() BlockStmt

		ASwitchStmt()
		Stmt
	}

	TypeSwitchStmt interface {
		Init() Stmt
		Assign() Stmt
		Body() BlockStmt

		ATypeSwitchStmt()
		Stmt
	}

	CommClause interface {
		Stmt
		ACommClause()

		Comm() Stmt
		Body() StmtList
	}

	SelectStmt interface {
		Stmt
		ASelectStmt()

		Body() BlockStmt
	}

	ForStmt interface {
		Init() Stmt
		Cond() Expr
		Post() Stmt
		Body() BlockStmt

		AForStmt()
		Stmt
	}

	RangeStmt interface {
		Stmt
		ARangeStmt()

		Key() Expr
		Value() Expr
		X() Expr
		Body() BlockStmt
		Tok() token.Token
		TokPos() Pos
	}
)

type StmtList interface {
	Len() int
	Stmt(i int) Stmt
}
