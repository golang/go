// UNREVIEWED
// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements syntax tree walking.
// TODO(gri) A more general API should probably be in
//           the syntax package.

package types2

import (
	"cmd/compile/internal/syntax"
	"fmt"
)

// Walk traverses a syntax in pre-order: It starts by calling f(root);
// root must not be nil. If f returns false (== "continue"), Walk calls
// f recursively for each of the non-nil children of that node; if f
// returns true (== "stop"), Walk does not traverse the respective node's
// children.
// Some nodes may be shared among multiple parent nodes (e.g., types in
// field lists such as type T in "a, b, c T"). Such shared nodes are
// walked multiple times.
// TODO(gri) Revisit this design. It may make sense to walk those nodes
//           only once. A place where this matters is TestResolveIdents.
func Walk(root syntax.Node, f func(syntax.Node) bool) {
	w := walker{f}
	w.node(root)
}

type walker struct {
	f func(syntax.Node) bool
}

func (w *walker) node(n syntax.Node) {
	if n == nil {
		panic("invalid syntax tree: nil node")
	}

	if w.f(n) {
		return
	}

	switch n := n.(type) {
	// packages
	case *syntax.File:
		w.node(n.PkgName)
		w.declList(n.DeclList)

	// declarations
	case *syntax.ImportDecl:
		if n.LocalPkgName != nil {
			w.node(n.LocalPkgName)
		}
		w.node(n.Path)

	case *syntax.ConstDecl:
		w.nameList(n.NameList)
		if n.Type != nil {
			w.node(n.Type)
		}
		if n.Values != nil {
			w.node(n.Values)
		}

	case *syntax.TypeDecl:
		w.node(n.Name)
		w.fieldList(n.TParamList)
		w.node(n.Type)

	case *syntax.VarDecl:
		w.nameList(n.NameList)
		if n.Type != nil {
			w.node(n.Type)
		}
		if n.Values != nil {
			w.node(n.Values)
		}

	case *syntax.FuncDecl:
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
	case *syntax.BadExpr: // nothing to do
	case *syntax.Name:
	case *syntax.BasicLit: // nothing to do

	case *syntax.CompositeLit:
		if n.Type != nil {
			w.node(n.Type)
		}
		w.exprList(n.ElemList)

	case *syntax.KeyValueExpr:
		w.node(n.Key)
		w.node(n.Value)

	case *syntax.FuncLit:
		w.node(n.Type)
		w.node(n.Body)

	case *syntax.ParenExpr:
		w.node(n.X)

	case *syntax.SelectorExpr:
		w.node(n.X)
		w.node(n.Sel)

	case *syntax.IndexExpr:
		w.node(n.X)
		w.node(n.Index)

	case *syntax.SliceExpr:
		w.node(n.X)
		for _, x := range n.Index {
			if x != nil {
				w.node(x)
			}
		}

	case *syntax.AssertExpr:
		w.node(n.X)
		w.node(n.Type)

	case *syntax.TypeSwitchGuard:
		if n.Lhs != nil {
			w.node(n.Lhs)
		}
		w.node(n.X)

	case *syntax.Operation:
		w.node(n.X)
		if n.Y != nil {
			w.node(n.Y)
		}

	case *syntax.CallExpr:
		w.node(n.Fun)
		w.exprList(n.ArgList)

	case *syntax.ListExpr:
		w.exprList(n.ElemList)

	// types
	case *syntax.ArrayType:
		if n.Len != nil {
			w.node(n.Len)
		}
		w.node(n.Elem)

	case *syntax.SliceType:
		w.node(n.Elem)

	case *syntax.DotsType:
		w.node(n.Elem)

	case *syntax.StructType:
		w.fieldList(n.FieldList)
		for _, t := range n.TagList {
			if t != nil {
				w.node(t)
			}
		}

	case *syntax.Field:
		if n.Name != nil {
			w.node(n.Name)
		}
		w.node(n.Type)

	case *syntax.InterfaceType:
		w.fieldList(n.MethodList)

	case *syntax.FuncType:
		w.fieldList(n.ParamList)
		w.fieldList(n.ResultList)

	case *syntax.MapType:
		w.node(n.Key)
		w.node(n.Value)

	case *syntax.ChanType:
		w.node(n.Elem)

	// statements
	case *syntax.EmptyStmt: // nothing to do

	case *syntax.LabeledStmt:
		w.node(n.Label)
		w.node(n.Stmt)

	case *syntax.BlockStmt:
		w.stmtList(n.List)

	case *syntax.ExprStmt:
		w.node(n.X)

	case *syntax.SendStmt:
		w.node(n.Chan)
		w.node(n.Value)

	case *syntax.DeclStmt:
		w.declList(n.DeclList)

	case *syntax.AssignStmt:
		w.node(n.Lhs)
		w.node(n.Rhs)

	case *syntax.BranchStmt:
		if n.Label != nil {
			w.node(n.Label)
		}
		// Target points to nodes elsewhere in the syntax tree

	case *syntax.CallStmt:
		w.node(n.Call)

	case *syntax.ReturnStmt:
		if n.Results != nil {
			w.node(n.Results)
		}

	case *syntax.IfStmt:
		if n.Init != nil {
			w.node(n.Init)
		}
		w.node(n.Cond)
		w.node(n.Then)
		if n.Else != nil {
			w.node(n.Else)
		}

	case *syntax.ForStmt:
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

	case *syntax.SwitchStmt:
		if n.Init != nil {
			w.node(n.Init)
		}
		if n.Tag != nil {
			w.node(n.Tag)
		}
		for _, s := range n.Body {
			w.node(s)
		}

	case *syntax.SelectStmt:
		for _, s := range n.Body {
			w.node(s)
		}

	// helper nodes
	case *syntax.RangeClause:
		if n.Lhs != nil {
			w.node(n.Lhs)
		}
		w.node(n.X)

	case *syntax.CaseClause:
		if n.Cases != nil {
			w.node(n.Cases)
		}
		w.stmtList(n.Body)

	case *syntax.CommClause:
		if n.Comm != nil {
			w.node(n.Comm)
		}
		w.stmtList(n.Body)

	default:
		panic(fmt.Sprintf("internal error: unknown node type %T", n))
	}
}

func (w *walker) declList(list []syntax.Decl) {
	for _, n := range list {
		w.node(n)
	}
}

func (w *walker) exprList(list []syntax.Expr) {
	for _, n := range list {
		w.node(n)
	}
}

func (w *walker) stmtList(list []syntax.Stmt) {
	for _, n := range list {
		w.node(n)
	}
}

func (w *walker) nameList(list []*syntax.Name) {
	for _, n := range list {
		w.node(n)
	}
}

func (w *walker) fieldList(list []*syntax.Field) {
	for _, n := range list {
		w.node(n)
	}
}
