// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

// ----------------------------------------------------------------------------
// Nodes

type Node interface {
	aNode()
}

type node struct {
	pos  uint32
	line uint32
	doc  *Comment // nil means no comment(s) attached
}

func (node) aNode() {}

func (n *node) init(p *parser) {
	n.pos = uint32(p.pos)
	n.line = uint32(p.line)
}

// ----------------------------------------------------------------------------
// Files

type File struct {
	node
	PkgName  *Name
	DeclList []Decl
	Lines    int
}

// ----------------------------------------------------------------------------
// Declarations

type (
	Decl interface {
		Node
		aDecl()
	}

	ImportDecl struct {
		decl
		LocalPkgName *Name // including "."; nil means no rename present
		Path         *BasicLit
		Group        *Group // nil means not part of a group
	}

	ConstDecl struct {
		decl
		NameList []*Name
		Type     Expr   // nil means no type
		Values   Expr   // nil means no values
		Group    *Group // nil means not part of a group
	}

	TypeDecl struct {
		decl
		Name  *Name
		Type  Expr
		Group *Group // nil means not part of a group
	}

	VarDecl struct {
		decl
		NameList []*Name
		Type     Expr   // nil means no type
		Values   Expr   // nil means no values
		Group    *Group // nil means not part of a group
	}

	FuncDecl struct {
		decl
		Attr map[string]bool // go:attr map
		Recv *Field          // nil means regular function
		Name *Name
		Type *FuncType
		Body []Stmt // nil means no body (forward declaration)
	}
)

type decl struct{ node }

func (decl) aDecl() {}

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

	// Value
	Name struct {
		expr
		Value string
	}

	// Value
	BasicLit struct {
		expr
		Value string
	}

	// Type { ElemList[0], ElemList[1], ... }
	CompositeLit struct {
		expr
		Type     Expr // nil means no literal type
		ElemList []Expr
		NKeys    int // number of elements with keys
	}

	// Key: Value
	KeyValueExpr struct {
		expr
		Key, Value Expr
	}

	// func Type { Body }
	FuncLit struct {
		expr
		Type *FuncType
		Body []Stmt
	}

	// (X)
	ParenExpr struct {
		expr
		X Expr
	}

	// X.Sel
	SelectorExpr struct {
		expr
		X   Expr
		Sel *Name
	}

	// X[Index]
	IndexExpr struct {
		expr
		X     Expr
		Index Expr
	}

	// X[Index[0] : Index[1] : Index[2]]
	SliceExpr struct {
		expr
		X     Expr
		Index [3]Expr
	}

	// X.(Type)
	AssertExpr struct {
		expr
		X Expr
		// TODO(gri) consider using Name{"..."} instead of nil (permits attaching of comments)
		Type Expr // nil means x.(type) (for use in type switch)
	}

	Operation struct {
		expr
		Op   Operator
		X, Y Expr // Y == nil means unary expression
	}

	// Fun(ArgList[0], ArgList[1], ...)
	CallExpr struct {
		expr
		Fun     Expr
		ArgList []Expr
		HasDots bool // last argument is followed by ...
	}

	// ElemList[0], ElemList[1], ...
	ListExpr struct {
		expr
		ElemList []Expr
	}

	// [Len]Elem
	ArrayType struct {
		expr
		// TODO(gri) consider using Name{"..."} instead of nil (permits attaching of comments)
		Len  Expr // nil means Len is ...
		Elem Expr
	}

	// []Elem
	SliceType struct {
		expr
		Elem Expr
	}

	// ...Elem
	DotsType struct {
		expr
		Elem Expr
	}

	// struct { FieldList[0] TagList[0]; FieldList[1] TagList[1]; ... }
	StructType struct {
		expr
		FieldList []*Field
		TagList   []*BasicLit // i >= len(TagList) || TagList[i] == nil means no tag for field i
	}

	// Name Type
	//      Type
	Field struct {
		node
		Name *Name // nil means anonymous field/parameter (structs/parameters), or embedded interface (interfaces)
		Type Expr  // field names declared in a list share the same Type (identical pointers)
	}

	// interface { MethodList[0]; MethodList[1]; ... }
	InterfaceType struct {
		expr
		MethodList []*Field
	}

	FuncType struct {
		expr
		ParamList  []*Field
		ResultList []*Field
	}

	// map[Key]Value
	MapType struct {
		expr
		Key   Expr
		Value Expr
	}

	//   chan Elem
	// <-chan Elem
	// chan<- Elem
	ChanType struct {
		expr
		Dir  ChanDir // 0 means no direction
		Elem Expr
	}
)

type expr struct{ node }

func (expr) aExpr() {}

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
		stmt
		Label *Name
		Stmt  Stmt
	}

	BlockStmt struct {
		stmt
		Body []Stmt
	}

	ExprStmt struct {
		simpleStmt
		X Expr
	}

	SendStmt struct {
		simpleStmt
		Chan, Value Expr // Chan <- Value
	}

	DeclStmt struct {
		stmt
		DeclList []Decl
	}

	AssignStmt struct {
		simpleStmt
		Op       Operator // 0 means no operation
		Lhs, Rhs Expr
	}

	BranchStmt struct {
		stmt
		Tok   token // TODO(gri) token values are not yet exported
		Label *Name
	}

	CallStmt struct {
		stmt
		Tok  token // _Go, or _Defer -- TODO(gri) token values are not yet exported
		Call *CallExpr
	}

	ReturnStmt struct {
		stmt
		Results Expr // nil means no (explicit) results
	}

	IfStmt struct {
		stmt
		Init SimpleStmt
		Cond Expr
		Then []Stmt
		Else []Stmt
	}

	ForStmt struct {
		stmt
		Init SimpleStmt // incl. *RangeClause
		Cond Expr
		Post SimpleStmt
		Body []Stmt
	}

	SwitchStmt struct {
		stmt
		Init SimpleStmt
		Tag  Expr
		Body []*CaseClause
	}

	SelectStmt struct {
		stmt
		Body []*CommClause
	}
)

type (
	RangeClause struct {
		simpleStmt
		Lhs Expr // nil means no Lhs = or Lhs :=
		Def bool // means :=
		X   Expr // range X
	}

	TypeSwitchGuard struct {
		expr
		// TODO(gri) consider using Name{"..."} instead of nil (permits attaching of comments)
		Lhs *Name // nil means no Lhs :=
		X   Expr  // X.(type)
	}

	CaseClause struct {
		node
		Cases Expr // nil means default clause
		Body  []Stmt
	}

	CommClause struct {
		node
		Comm SimpleStmt // send or receive stmt; nil means default clause
		Body []Stmt
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
