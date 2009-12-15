// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ast

import "fmt"

// A Visitor's Visit method is invoked for each node encountered by Walk.
// If the result visitor w is not nil, Walk visits each of the children
// of node with the visitor w, followed by a call of w.Visit(nil).
type Visitor interface {
	Visit(node interface{}) (w Visitor)
}


func walkIdent(v Visitor, x *Ident) {
	if x != nil {
		Walk(v, x)
	}
}


func walkCommentGroup(v Visitor, g *CommentGroup) {
	if g != nil {
		Walk(v, g)
	}
}


func walkBlockStmt(v Visitor, b *BlockStmt) {
	if b != nil {
		Walk(v, b)
	}
}


// Walk traverses an AST in depth-first order: If node != nil, it
// invokes v.Visit(node). If the visitor w returned by v.Visit(node) is
// not nil, Walk visits each of the children of node with the visitor w,
// followed by a call of w.Visit(nil).
//
// Walk may be called with any of the named ast node types. It also
// accepts arguments of type []*Field, []*Ident, []Expr and []Stmt;
// the respective children are the slice elements.
//
func Walk(v Visitor, node interface{}) {
	if node == nil {
		return
	}
	if v = v.Visit(node); v == nil {
		return
	}

	// walk children
	// (the order of the cases matches the order
	// of the corresponding declaration in ast.go)
	switch n := node.(type) {
	// Comments and fields
	case *Comment:
		// nothing to do

	case *CommentGroup:
		for _, c := range n.List {
			Walk(v, c)
		}
		// TODO(gri): Keep comments in a list/vector instead
		// of linking them via Next. Following next will lead
		// to multiple visits and potentially n^2 behavior
		// since Doc and Comments fields point into the global
		// comments list.

	case *Field:
		walkCommentGroup(v, n.Doc)
		Walk(v, n.Names)
		Walk(v, n.Type)
		for _, x := range n.Tag {
			Walk(v, x)
		}
		walkCommentGroup(v, n.Comment)

	// Expressions
	case *BadExpr, *Ident, *Ellipsis, *BasicLit:
		// nothing to do

	case *StringList:
		for _, x := range n.Strings {
			Walk(v, x)
		}

	case *FuncLit:
		if n != nil {
			Walk(v, n.Type)
		}
		walkBlockStmt(v, n.Body)

	case *CompositeLit:
		Walk(v, n.Type)
		Walk(v, n.Elts)

	case *ParenExpr:
		Walk(v, n.X)

	case *SelectorExpr:
		Walk(v, n.X)
		walkIdent(v, n.Sel)

	case *IndexExpr:
		Walk(v, n.X)
		Walk(v, n.Index)

	case *SliceExpr:
		Walk(v, n.X)
		Walk(v, n.Index)
		Walk(v, n.End)

	case *TypeAssertExpr:
		Walk(v, n.X)
		Walk(v, n.Type)

	case *CallExpr:
		Walk(v, n.Fun)
		Walk(v, n.Args)

	case *StarExpr:
		Walk(v, n.X)

	case *UnaryExpr:
		Walk(v, n.X)

	case *BinaryExpr:
		Walk(v, n.X)
		Walk(v, n.Y)

	case *KeyValueExpr:
		Walk(v, n.Key)
		Walk(v, n.Value)

	// Types
	case *ArrayType:
		Walk(v, n.Len)
		Walk(v, n.Elt)

	case *StructType:
		Walk(v, n.Fields)

	case *FuncType:
		Walk(v, n.Params)
		Walk(v, n.Results)

	case *InterfaceType:
		Walk(v, n.Methods)

	case *MapType:
		Walk(v, n.Key)
		Walk(v, n.Value)

	case *ChanType:
		Walk(v, n.Value)

	// Statements
	case *BadStmt:
		// nothing to do

	case *DeclStmt:
		Walk(v, n.Decl)

	case *EmptyStmt:
		// nothing to do

	case *LabeledStmt:
		walkIdent(v, n.Label)
		Walk(v, n.Stmt)

	case *ExprStmt:
		Walk(v, n.X)

	case *IncDecStmt:
		Walk(v, n.X)

	case *AssignStmt:
		Walk(v, n.Lhs)
		Walk(v, n.Rhs)

	case *GoStmt:
		if n.Call != nil {
			Walk(v, n.Call)
		}

	case *DeferStmt:
		if n.Call != nil {
			Walk(v, n.Call)
		}

	case *ReturnStmt:
		Walk(v, n.Results)

	case *BranchStmt:
		walkIdent(v, n.Label)

	case *BlockStmt:
		Walk(v, n.List)

	case *IfStmt:
		Walk(v, n.Init)
		Walk(v, n.Cond)
		walkBlockStmt(v, n.Body)
		Walk(v, n.Else)

	case *CaseClause:
		Walk(v, n.Values)
		Walk(v, n.Body)

	case *SwitchStmt:
		Walk(v, n.Init)
		Walk(v, n.Tag)
		walkBlockStmt(v, n.Body)

	case *TypeCaseClause:
		Walk(v, n.Types)
		Walk(v, n.Body)

	case *TypeSwitchStmt:
		Walk(v, n.Init)
		Walk(v, n.Assign)
		walkBlockStmt(v, n.Body)

	case *CommClause:
		Walk(v, n.Lhs)
		Walk(v, n.Rhs)
		Walk(v, n.Body)

	case *SelectStmt:
		walkBlockStmt(v, n.Body)

	case *ForStmt:
		Walk(v, n.Init)
		Walk(v, n.Cond)
		Walk(v, n.Post)
		walkBlockStmt(v, n.Body)

	case *RangeStmt:
		Walk(v, n.Key)
		Walk(v, n.Value)
		Walk(v, n.X)
		walkBlockStmt(v, n.Body)

	// Declarations
	case *ImportSpec:
		walkCommentGroup(v, n.Doc)
		walkIdent(v, n.Name)
		for _, x := range n.Path {
			Walk(v, x)
		}
		walkCommentGroup(v, n.Comment)

	case *ValueSpec:
		walkCommentGroup(v, n.Doc)
		Walk(v, n.Names)
		Walk(v, n.Type)
		Walk(v, n.Values)
		walkCommentGroup(v, n.Comment)

	case *TypeSpec:
		walkCommentGroup(v, n.Doc)
		walkIdent(v, n.Name)
		Walk(v, n.Type)
		walkCommentGroup(v, n.Comment)

	case *BadDecl:
		// nothing to do

	case *GenDecl:
		walkCommentGroup(v, n.Doc)
		for _, s := range n.Specs {
			Walk(v, s)
		}

	case *FuncDecl:
		walkCommentGroup(v, n.Doc)
		if n.Recv != nil {
			Walk(v, n.Recv)
		}
		walkIdent(v, n.Name)
		if n.Type != nil {
			Walk(v, n.Type)
		}
		walkBlockStmt(v, n.Body)

	// Files and packages
	case *File:
		walkCommentGroup(v, n.Doc)
		walkIdent(v, n.Name)
		for _, d := range n.Decls {
			Walk(v, d)
		}
		walkCommentGroup(v, n.Comments)

	case *Package:
		for _, f := range n.Files {
			Walk(v, f)
		}

	case []*Field:
		for _, x := range n {
			Walk(v, x)
		}

	case []*Ident:
		for _, x := range n {
			Walk(v, x)
		}

	case []Expr:
		for _, x := range n {
			Walk(v, x)
		}

	case []Stmt:
		for _, x := range n {
			Walk(v, x)
		}

	default:
		fmt.Printf("ast.Walk: unexpected type %T", n)
		panic()
	}

	v.Visit(nil)
}
