// UNREVIEWED
// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements helper functions for scope position computations.

package types2

import "cmd/compile/internal/syntax"

// startPos returns the start position of n.
func startPos(n syntax.Node) syntax.Pos {
	// Cases for nodes which don't need a correction are commented out.
	for m := n; ; {
		switch n := m.(type) {
		case nil:
			panic("internal error: nil")

		// packages
		case *syntax.File:
			// file block starts at the beginning of the file
			return syntax.MakePos(n.Pos().Base(), 1, 1)

		// declarations
		// case *syntax.ImportDecl:
		// case *syntax.ConstDecl:
		// case *syntax.TypeDecl:
		// case *syntax.VarDecl:
		// case *syntax.FuncDecl:

		// expressions
		// case *syntax.BadExpr:
		// case *syntax.Name:
		// case *syntax.BasicLit:
		case *syntax.CompositeLit:
			if n.Type != nil {
				m = n.Type
				continue
			}
			return n.Pos()
		// case *syntax.KeyValueExpr:
		// case *syntax.FuncLit:
		// case *syntax.ParenExpr:
		case *syntax.SelectorExpr:
			m = n.X
		case *syntax.IndexExpr:
			m = n.X
		// case *syntax.SliceExpr:
		case *syntax.AssertExpr:
			m = n.X
		case *syntax.TypeSwitchGuard:
			if n.Lhs != nil {
				m = n.Lhs
				continue
			}
			m = n.X
		case *syntax.Operation:
			if n.Y != nil {
				m = n.X
				continue
			}
			return n.Pos()
		case *syntax.CallExpr:
			m = n.Fun
		case *syntax.ListExpr:
			if len(n.ElemList) > 0 {
				m = n.ElemList[0]
				continue
			}
			return n.Pos()
		// types
		// case *syntax.ArrayType:
		// case *syntax.SliceType:
		// case *syntax.DotsType:
		// case *syntax.StructType:
		// case *syntax.Field:
		// case *syntax.InterfaceType:
		// case *syntax.FuncType:
		// case *syntax.MapType:
		// case *syntax.ChanType:

		// statements
		// case *syntax.EmptyStmt:
		// case *syntax.LabeledStmt:
		// case *syntax.BlockStmt:
		// case *syntax.ExprStmt:
		case *syntax.SendStmt:
			m = n.Chan
		// case *syntax.DeclStmt:
		case *syntax.AssignStmt:
			m = n.Lhs
		// case *syntax.BranchStmt:
		// case *syntax.CallStmt:
		// case *syntax.ReturnStmt:
		// case *syntax.IfStmt:
		// case *syntax.ForStmt:
		// case *syntax.SwitchStmt:
		// case *syntax.SelectStmt:

		// helper nodes
		case *syntax.RangeClause:
			if n.Lhs != nil {
				m = n.Lhs
				continue
			}
			m = n.X
		// case *syntax.CaseClause:
		// case *syntax.CommClause:

		default:
			return n.Pos()
		}
	}
}

// endPos returns the approximate end position of n in the source.
// For some nodes (*syntax.Name, *syntax.BasicLit) it returns
// the position immediately following the node; for others
// (*syntax.BlockStmt, *syntax.SwitchStmt, etc.) it returns
// the position of the closing '}'; and for some (*syntax.ParenExpr)
// the returned position is the end position of the last enclosed
// expression.
// Thus, endPos should not be used for exact demarcation of the
// end of a node in the source; it is mostly useful to determine
// scope ranges where there is some leeway.
func endPos(n syntax.Node) syntax.Pos {
	for m := n; ; {
		switch n := m.(type) {
		case nil:
			panic("internal error: nil")

		// packages
		case *syntax.File:
			return n.EOF

		// declarations
		case *syntax.ImportDecl:
			m = n.Path
		case *syntax.ConstDecl:
			if n.Values != nil {
				m = n.Values
				continue
			}
			if n.Type != nil {
				m = n.Type
				continue
			}
			if l := len(n.NameList); l > 0 {
				m = n.NameList[l-1]
				continue
			}
			return n.Pos()
		case *syntax.TypeDecl:
			m = n.Type
		case *syntax.VarDecl:
			if n.Values != nil {
				m = n.Values
				continue
			}
			if n.Type != nil {
				m = n.Type
				continue
			}
			if l := len(n.NameList); l > 0 {
				m = n.NameList[l-1]
				continue
			}
			return n.Pos()
		case *syntax.FuncDecl:
			if n.Body != nil {
				m = n.Body
				continue
			}
			m = n.Type

		// expressions
		case *syntax.BadExpr:
			return n.Pos()
		case *syntax.Name:
			p := n.Pos()
			return syntax.MakePos(p.Base(), p.Line(), p.Col()+uint(len(n.Value)))
		case *syntax.BasicLit:
			p := n.Pos()
			return syntax.MakePos(p.Base(), p.Line(), p.Col()+uint(len(n.Value)))
		case *syntax.CompositeLit:
			return n.Rbrace
		case *syntax.KeyValueExpr:
			m = n.Value
		case *syntax.FuncLit:
			m = n.Body
		case *syntax.ParenExpr:
			m = n.X
		case *syntax.SelectorExpr:
			m = n.Sel
		case *syntax.IndexExpr:
			m = n.Index
		case *syntax.SliceExpr:
			for i := len(n.Index) - 1; i >= 0; i-- {
				if x := n.Index[i]; x != nil {
					m = x
					continue
				}
			}
			m = n.X
		case *syntax.AssertExpr:
			m = n.Type
		case *syntax.TypeSwitchGuard:
			m = n.X
		case *syntax.Operation:
			if n.Y != nil {
				m = n.Y
				continue
			}
			m = n.X
		case *syntax.CallExpr:
			if l := lastExpr(n.ArgList); l != nil {
				m = l
				continue
			}
			m = n.Fun
		case *syntax.ListExpr:
			if l := lastExpr(n.ElemList); l != nil {
				m = l
				continue
			}
			return n.Pos()

		// types
		case *syntax.ArrayType:
			m = n.Elem
		case *syntax.SliceType:
			m = n.Elem
		case *syntax.DotsType:
			m = n.Elem
		case *syntax.StructType:
			if l := lastField(n.FieldList); l != nil {
				m = l
				continue
			}
			return n.Pos()
			// TODO(gri) need to take TagList into account
		case *syntax.Field:
			if n.Type != nil {
				m = n.Type
				continue
			}
			m = n.Name
		case *syntax.InterfaceType:
			if l := lastField(n.MethodList); l != nil {
				m = l
				continue
			}
			return n.Pos()
		case *syntax.FuncType:
			if l := lastField(n.ResultList); l != nil {
				m = l
				continue
			}
			if l := lastField(n.ParamList); l != nil {
				m = l
				continue
			}
			return n.Pos()
		case *syntax.MapType:
			m = n.Value
		case *syntax.ChanType:
			m = n.Elem

		// statements
		case *syntax.EmptyStmt:
			return n.Pos()
		case *syntax.LabeledStmt:
			m = n.Stmt
		case *syntax.BlockStmt:
			return n.Rbrace
		case *syntax.ExprStmt:
			m = n.X
		case *syntax.SendStmt:
			m = n.Value
		case *syntax.DeclStmt:
			if l := lastDecl(n.DeclList); l != nil {
				m = l
				continue
			}
			return n.Pos()
		case *syntax.AssignStmt:
			m = n.Rhs
			if m == nil {
				p := endPos(n.Lhs)
				return syntax.MakePos(p.Base(), p.Line(), p.Col()+2)
			}
		case *syntax.BranchStmt:
			if n.Label != nil {
				m = n.Label
				continue
			}
			return n.Pos()
		case *syntax.CallStmt:
			m = n.Call
		case *syntax.ReturnStmt:
			if n.Results != nil {
				m = n.Results
				continue
			}
			return n.Pos()
		case *syntax.IfStmt:
			if n.Else != nil {
				m = n.Else
				continue
			}
			m = n.Then
		case *syntax.ForStmt:
			m = n.Body
		case *syntax.SwitchStmt:
			return n.Rbrace
		case *syntax.SelectStmt:
			return n.Rbrace

		// helper nodes
		case *syntax.RangeClause:
			m = n.X
		case *syntax.CaseClause:
			if l := lastStmt(n.Body); l != nil {
				m = l
				continue
			}
			return n.Colon
		case *syntax.CommClause:
			if l := lastStmt(n.Body); l != nil {
				m = l
				continue
			}
			return n.Colon

		default:
			return n.Pos()
		}
	}
}

func lastDecl(list []syntax.Decl) syntax.Decl {
	if l := len(list); l > 0 {
		return list[l-1]
	}
	return nil
}

func lastExpr(list []syntax.Expr) syntax.Expr {
	if l := len(list); l > 0 {
		return list[l-1]
	}
	return nil
}

func lastStmt(list []syntax.Stmt) syntax.Stmt {
	if l := len(list); l > 0 {
		return list[l-1]
	}
	return nil
}

func lastField(list []*syntax.Field) *syntax.Field {
	if l := len(list); l > 0 {
		return list[l-1]
	}
	return nil
}
