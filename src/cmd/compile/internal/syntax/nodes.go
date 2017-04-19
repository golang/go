// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import "cmd/internal/src"

// ----------------------------------------------------------------------------
// Nodes

type Node interface {
	// Pos() returns the position associated with the node as follows:
	// 1) The position of a node representing a terminal syntax production
	//    (Name, BasicLit, etc.) is the position of the respective production
	//    in the source.
	// 2) The position of a node representing a non-terminal production
	//    (IndexExpr, IfStmt, etc.) is the position of a token uniquely
	//    associated with that production; usually the left-most one
	//    ('[' for IndexExpr, 'if' for IfStmt, etc.)
	Pos() src.Pos
	aNode()
}

type node struct {
	// commented out for now since not yet used
	// doc  *Comment // nil means no comment(s) attached
	pos src.Pos
}

func (n *node) Pos() src.Pos { return n.pos }
func (*node) aNode()         {}

// ----------------------------------------------------------------------------
// Files

// package PkgName; DeclList[0], DeclList[1], ...
type File struct {
	PkgName  *Name
	DeclList []Decl
	Lines    uint
	node
}

// ----------------------------------------------------------------------------
// Declarations

type (
	Decl interface {
		Node
		aDecl()
	}

	//              Path
	// LocalPkgName Path
	ImportDecl struct {
		LocalPkgName *Name // including "."; nil means no rename present
		Path         *BasicLit
		Group        *Group // nil means not part of a group
		decl
	}

	// NameList
	// NameList      = Values
	// NameList Type = Values
	ConstDecl struct {
		NameList []*Name
		Type     Expr   // nil means no type
		Values   Expr   // nil means no values
		Group    *Group // nil means not part of a group
		decl
	}

	// Name Type
	TypeDecl struct {
		Name   *Name
		Alias  bool
		Type   Expr
		Group  *Group // nil means not part of a group
		Pragma Pragma
		decl
	}

	// NameList Type
	// NameList Type = Values
	// NameList      = Values
	VarDecl struct {
		NameList []*Name
		Type     Expr   // nil means no type
		Values   Expr   // nil means no values
		Group    *Group // nil means not part of a group
		decl
	}

	// func          Name Type { Body }
	// func          Name Type
	// func Receiver Name Type { Body }
	// func Receiver Name Type
	FuncDecl struct {
		Attr   map[string]bool // go:attr map
		Recv   *Field          // nil means regular function
		Name   *Name
		Type   *FuncType
		Body   *BlockStmt // nil means no body (forward declaration)
		Pragma Pragma     // TODO(mdempsky): Cleaner solution.
		decl
	}
)

type decl struct{ node }

func (*decl) aDecl() {}

// All declarations belonging to the same group point to the same Group node.
type Group struct {
	dummy int // not empty so we are guaranteed different Group instances
}

// ----------------------------------------------------------------------------
// Expressions

type (
	Expr interface {
		Node
		aExpr()
	}

	// Placeholder for an expression that failed to parse
	// correctly and where we can't provide a better node.
	BadExpr struct {
		expr
	}

	// Value
	Name struct {
		Value string
		expr
	}

	// Value
	BasicLit struct {
		Value string
		Kind  LitKind
		expr
	}

	// Type { ElemList[0], ElemList[1], ... }
	CompositeLit struct {
		Type     Expr // nil means no literal type
		ElemList []Expr
		NKeys    int // number of elements with keys
		Rbrace   src.Pos
		expr
	}

	// Key: Value
	KeyValueExpr struct {
		Key, Value Expr
		expr
	}

	// func Type { Body }
	FuncLit struct {
		Type *FuncType
		Body *BlockStmt
		expr
	}

	// (X)
	ParenExpr struct {
		X Expr
		expr
	}

	// X.Sel
	SelectorExpr struct {
		X   Expr
		Sel *Name
		expr
	}

	// X[Index]
	IndexExpr struct {
		X     Expr
		Index Expr
		expr
	}

	// X[Index[0] : Index[1] : Index[2]]
	SliceExpr struct {
		X     Expr
		Index [3]Expr
		// Full indicates whether this is a simple or full slice expression.
		// In a valid AST, this is equivalent to Index[2] != nil.
		// TODO(mdempsky): This is only needed to report the "3-index
		// slice of string" error when Index[2] is missing.
		Full bool
		expr
	}

	// X.(Type)
	AssertExpr struct {
		X Expr
		// TODO(gri) consider using Name{"..."} instead of nil (permits attaching of comments)
		Type Expr
		expr
	}

	Operation struct {
		Op   Operator
		X, Y Expr // Y == nil means unary expression
		expr
	}

	// Fun(ArgList[0], ArgList[1], ...)
	CallExpr struct {
		Fun     Expr
		ArgList []Expr
		HasDots bool // last argument is followed by ...
		expr
	}

	// ElemList[0], ElemList[1], ...
	ListExpr struct {
		ElemList []Expr
		expr
	}

	// [Len]Elem
	ArrayType struct {
		// TODO(gri) consider using Name{"..."} instead of nil (permits attaching of comments)
		Len  Expr // nil means Len is ...
		Elem Expr
		expr
	}

	// []Elem
	SliceType struct {
		Elem Expr
		expr
	}

	// ...Elem
	DotsType struct {
		Elem Expr
		expr
	}

	// struct { FieldList[0] TagList[0]; FieldList[1] TagList[1]; ... }
	StructType struct {
		FieldList []*Field
		TagList   []*BasicLit // i >= len(TagList) || TagList[i] == nil means no tag for field i
		expr
	}

	// Name Type
	//      Type
	Field struct {
		Name *Name // nil means anonymous field/parameter (structs/parameters), or embedded interface (interfaces)
		Type Expr  // field names declared in a list share the same Type (identical pointers)
		node
	}

	// interface { MethodList[0]; MethodList[1]; ... }
	InterfaceType struct {
		MethodList []*Field
		expr
	}

	FuncType struct {
		ParamList  []*Field
		ResultList []*Field
		expr
	}

	// map[Key]Value
	MapType struct {
		Key   Expr
		Value Expr
		expr
	}

	//   chan Elem
	// <-chan Elem
	// chan<- Elem
	ChanType struct {
		Dir  ChanDir // 0 means no direction
		Elem Expr
		expr
	}
)

type expr struct{ node }

func (*expr) aExpr() {}

type ChanDir uint

const (
	_ ChanDir = iota
	SendOnly
	RecvOnly
)

// ----------------------------------------------------------------------------
// Statements

type (
	Stmt interface {
		Node
		aStmt()
	}

	SimpleStmt interface {
		Stmt
		aSimpleStmt()
	}

	EmptyStmt struct {
		simpleStmt
	}

	LabeledStmt struct {
		Label *Name
		Stmt  Stmt
		stmt
	}

	BlockStmt struct {
		List   []Stmt
		Rbrace src.Pos
		stmt
	}

	ExprStmt struct {
		X Expr
		simpleStmt
	}

	SendStmt struct {
		Chan, Value Expr // Chan <- Value
		simpleStmt
	}

	DeclStmt struct {
		DeclList []Decl
		stmt
	}

	AssignStmt struct {
		Op       Operator // 0 means no operation
		Lhs, Rhs Expr     // Rhs == ImplicitOne means Lhs++ (Op == Add) or Lhs-- (Op == Sub)
		simpleStmt
	}

	BranchStmt struct {
		Tok   token // Break, Continue, Fallthrough, or Goto
		Label *Name
		// Target is the continuation of the control flow after executing
		// the branch; it is computed by the parser if CheckBranches is set.
		// Target is a *LabeledStmt for gotos, and a *SwitchStmt, *SelectStmt,
		// or *ForStmt for breaks and continues, depending on the context of
		// the branch. Target is not set for fallthroughs.
		Target Stmt
		stmt
	}

	CallStmt struct {
		Tok  token // Go or Defer
		Call *CallExpr
		stmt
	}

	ReturnStmt struct {
		Results Expr // nil means no explicit return values
		stmt
	}

	IfStmt struct {
		Init SimpleStmt
		Cond Expr
		Then *BlockStmt
		Else Stmt // either *IfStmt or *BlockStmt
		stmt
	}

	ForStmt struct {
		Init SimpleStmt // incl. *RangeClause
		Cond Expr
		Post SimpleStmt
		Body *BlockStmt
		stmt
	}

	SwitchStmt struct {
		Init   SimpleStmt
		Tag    Expr
		Body   []*CaseClause
		Rbrace src.Pos
		stmt
	}

	SelectStmt struct {
		Body   []*CommClause
		Rbrace src.Pos
		stmt
	}
)

type (
	RangeClause struct {
		Lhs Expr // nil means no Lhs = or Lhs :=
		Def bool // means :=
		X   Expr // range X
		simpleStmt
	}

	TypeSwitchGuard struct {
		// TODO(gri) consider using Name{"..."} instead of nil (permits attaching of comments)
		Lhs *Name // nil means no Lhs :=
		X   Expr  // X.(type)
		expr
	}

	CaseClause struct {
		Cases Expr // nil means default clause
		Body  []Stmt
		Colon src.Pos
		node
	}

	CommClause struct {
		Comm  SimpleStmt // send or receive stmt; nil means default clause
		Body  []Stmt
		Colon src.Pos
		node
	}
)

type stmt struct{ node }

func (stmt) aStmt() {}

type simpleStmt struct {
	stmt
}

func (simpleStmt) aSimpleStmt() {}

// ----------------------------------------------------------------------------
// Comments

// TODO(gri) Consider renaming to CommentPos, CommentPlacement, etc.
//           Kind = Above doesn't make much sense.
type CommentKind uint

const (
	Above CommentKind = iota
	Below
	Left
	Right
)

type Comment struct {
	Kind CommentKind
	Text string
	Next *Comment
}
