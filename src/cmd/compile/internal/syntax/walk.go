// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements syntax tree walking.

package syntax

import "fmt"

// Inspect traverses an AST in pre-order: It starts by calling
// f(node); node must not be nil. If f returns true, Inspect invokes f
// recursively for each of the non-nil children of node, followed by a
// call of f(nil).
//
// See Walk for caveats about shared nodes.
func Inspect(root Node, f func(Node) bool) {
	Walk(root, inspector(f))
}

type inspector func(Node) bool

func (v inspector) Visit(node Node) Visitor {
	if v(node) {
		return v
	}
	return nil
}

// Crawl traverses a syntax in pre-order: It starts by calling f(root);
// root must not be nil. If f returns false (== "continue"), Crawl calls
// f recursively for each of the non-nil children of that node; if f
// returns true (== "stop"), Crawl does not traverse the respective node's
// children.
//
// See Walk for caveats about shared nodes.
//
// Deprecated: Use Inspect instead.
func Crawl(root Node, f func(Node) bool) {
	Inspect(root, func(node Node) bool {
		return node != nil && !f(node)
	})
}

// Walk traverses an AST in pre-order: It starts by calling
// v.Visit(node); node must not be nil. If the visitor w returned by
// v.Visit(node) is not nil, Walk is invoked recursively with visitor
// w for each of the non-nil children of node, followed by a call of
// w.Visit(nil).
//
// Some nodes may be shared among multiple parent nodes (e.g., types in
// field lists such as type T in "a, b, c T"). Such shared nodes are
// walked multiple times.
// TODO(gri) Revisit this design. It may make sense to walk those nodes
//           only once. A place where this matters is types2.TestResolveIdents.
func Walk(root Node, v Visitor) {
	walker{v}.node(root)
}

// A Visitor's Visit method is invoked for each node encountered by Walk.
// If the result visitor w is not nil, Walk visits each of the children
// of node with the visitor w, followed by a call of w.Visit(nil).
type Visitor interface {
	Visit(node Node) (w Visitor)
}

type walker struct {
	v Visitor
}

func (w walker) node(n Node) {
	if n == nil {
		panic("nil node")
	}

	w.v = w.v.Visit(n)
	if w.v == nil {
		return
	}

	switch n := n.(type) {
	// packages
	case *File:
		w.node(n.PkgName)
		w.declList(n.DeclList)

	// declarations
	case *ImportDecl:
		if n.LocalPkgName != nil {
			w.node(n.LocalPkgName)
		}
		w.node(n.Path)

	case *ConstDecl:
		w.nameList(n.NameList)
		if n.Type != nil {
			w.node(n.Type)
		}
		if n.Values != nil {
			w.node(n.Values)
		}

	case *TypeDecl:
		w.node(n.Name)
		w.fieldList(n.TParamList)
		w.node(n.Type)

	case *VarDecl:
		w.nameList(n.NameList)
		if n.Type != nil {
			w.node(n.Type)
		}
		if n.Values != nil {
			w.node(n.Values)
		}

	case *FuncDecl:
		if n.Recv != nil {
			w.node(n.Recv)
		}
		w.node(n.Name)
		w.fieldList(n.TParamList)
		w.node(n.Type)
		if n.Body != nil {
			w.node(n.Body)
		}

	// expressions
	case *BadExpr: // nothing to do
	case *Name: // nothing to do
	case *BasicLit: // nothing to do

	case *CompositeLit:
		if n.Type != nil {
			w.node(n.Type)
		}
		w.exprList(n.ElemList)

	case *KeyValueExpr:
		w.node(n.Key)
		w.node(n.Value)

	case *FuncLit:
		w.node(n.Type)
		w.node(n.Body)

	case *ParenExpr:
		w.node(n.X)

	case *SelectorExpr:
		w.node(n.X)
		w.node(n.Sel)

	case *IndexExpr:
		w.node(n.X)
		w.node(n.Index)

	case *SliceExpr:
		w.node(n.X)
		for _, x := range n.Index {
			if x != nil {
				w.node(x)
			}
		}

	case *AssertExpr:
		w.node(n.X)
		w.node(n.Type)

	case *TypeSwitchGuard:
		if n.Lhs != nil {
			w.node(n.Lhs)
		}
		w.node(n.X)

	case *Operation:
		w.node(n.X)
		if n.Y != nil {
			w.node(n.Y)
		}

	case *CallExpr:
		w.node(n.Fun)
		w.exprList(n.ArgList)

	case *ListExpr:
		w.exprList(n.ElemList)

	// types
	case *ArrayType:
		if n.Len != nil {
			w.node(n.Len)
		}
		w.node(n.Elem)

	case *SliceType:
		w.node(n.Elem)

	case *DotsType:
		w.node(n.Elem)

	case *StructType:
		w.fieldList(n.FieldList)
		for _, t := range n.TagList {
			if t != nil {
				w.node(t)
			}
		}

	case *Field:
		if n.Name != nil {
			w.node(n.Name)
		}
		w.node(n.Type)

	case *InterfaceType:
		w.fieldList(n.MethodList)

	case *FuncType:
		w.fieldList(n.ParamList)
		w.fieldList(n.ResultList)

	case *MapType:
		w.node(n.Key)
		w.node(n.Value)

	case *ChanType:
		w.node(n.Elem)

	// statements
	case *EmptyStmt: // nothing to do

	case *LabeledStmt:
		w.node(n.Label)
		w.node(n.Stmt)

	case *BlockStmt:
		w.stmtList(n.List)

	case *ExprStmt:
		w.node(n.X)

	case *SendStmt:
		w.node(n.Chan)
		w.node(n.Value)

	case *DeclStmt:
		w.declList(n.DeclList)

	case *AssignStmt:
		w.node(n.Lhs)
		if n.Rhs != nil {
			w.node(n.Rhs)
		}

	case *BranchStmt:
		if n.Label != nil {
			w.node(n.Label)
		}
		// Target points to nodes elsewhere in the syntax tree

	case *CallStmt:
		w.node(n.Call)

	case *ReturnStmt:
		if n.Results != nil {
			w.node(n.Results)
		}

	case *IfStmt:
		if n.Init != nil {
			w.node(n.Init)
		}
		w.node(n.Cond)
		w.node(n.Then)
		if n.Else != nil {
			w.node(n.Else)
		}

	case *ForStmt:
		if n.Init != nil {
			w.node(n.Init)
		}
		if n.Cond != nil {
			w.node(n.Cond)
		}
		if n.Post != nil {
			w.node(n.Post)
		}
		w.node(n.Body)

	case *SwitchStmt:
		if n.Init != nil {
			w.node(n.Init)
		}
		if n.Tag != nil {
			w.node(n.Tag)
		}
		for _, s := range n.Body {
			w.node(s)
		}

	case *SelectStmt:
		for _, s := range n.Body {
			w.node(s)
		}

	// helper nodes
	case *RangeClause:
		if n.Lhs != nil {
			w.node(n.Lhs)
		}
		w.node(n.X)

	case *CaseClause:
		if n.Cases != nil {
			w.node(n.Cases)
		}
		w.stmtList(n.Body)

	case *CommClause:
		if n.Comm != nil {
			w.node(n.Comm)
		}
		w.stmtList(n.Body)

	default:
		panic(fmt.Sprintf("internal error: unknown node type %T", n))
	}

	w.v.Visit(nil)
}

func (w walker) declList(list []Decl) {
	for _, n := range list {
		w.node(n)
	}
}

func (w walker) exprList(list []Expr) {
	for _, n := range list {
		w.node(n)
	}
}

func (w walker) stmtList(list []Stmt) {
	for _, n := range list {
		w.node(n)
	}
}

func (w walker) nameList(list []*Name) {
	for _, n := range list {
		w.node(n)
	}
}

func (w walker) fieldList(list []*Field) {
	for _, n := range list {
		w.node(n)
	}
}
