// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package inspector

// This file is a fork of ast.Inspect to reduce unnecessary dynamic
// calls and to gather edge information.
//
// Consistency with the original is ensured by TestInspectAllNodes.

import (
	"fmt"
	"go/ast"

	"golang.org/x/tools/go/ast/edge"
)

func walkList[N ast.Node](v *visitor, ek edge.Kind, list []N) {
	for i, node := range list {
		walk(v, ek, i, node)
	}
}

func walk(v *visitor, ek edge.Kind, index int, node ast.Node) {
	v.push(ek, index, node)

	// walk children
	// (the order of the cases matches the order
	// of the corresponding node types in ast.go)
	switch n := node.(type) {
	// Comments and fields
	case *ast.Comment:
		// nothing to do

	case *ast.CommentGroup:
		walkList(v, edge.CommentGroup_List, n.List)

	case *ast.Field:
		if n.Doc != nil {
			walk(v, edge.Field_Doc, -1, n.Doc)
		}
		walkList(v, edge.Field_Names, n.Names)
		if n.Type != nil {
			walk(v, edge.Field_Type, -1, n.Type)
		}
		if n.Tag != nil {
			walk(v, edge.Field_Tag, -1, n.Tag)
		}
		if n.Comment != nil {
			walk(v, edge.Field_Comment, -1, n.Comment)
		}

	case *ast.FieldList:
		walkList(v, edge.FieldList_List, n.List)

	// Expressions
	case *ast.BadExpr, *ast.Ident, *ast.BasicLit:
		// nothing to do

	case *ast.Ellipsis:
		if n.Elt != nil {
			walk(v, edge.Ellipsis_Elt, -1, n.Elt)
		}

	case *ast.FuncLit:
		walk(v, edge.FuncLit_Type, -1, n.Type)
		walk(v, edge.FuncLit_Body, -1, n.Body)

	case *ast.CompositeLit:
		if n.Type != nil {
			walk(v, edge.CompositeLit_Type, -1, n.Type)
		}
		walkList(v, edge.CompositeLit_Elts, n.Elts)

	case *ast.ParenExpr:
		walk(v, edge.ParenExpr_X, -1, n.X)

	case *ast.SelectorExpr:
		walk(v, edge.SelectorExpr_X, -1, n.X)
		walk(v, edge.SelectorExpr_Sel, -1, n.Sel)

	case *ast.IndexExpr:
		walk(v, edge.IndexExpr_X, -1, n.X)
		walk(v, edge.IndexExpr_Index, -1, n.Index)

	case *ast.IndexListExpr:
		walk(v, edge.IndexListExpr_X, -1, n.X)
		walkList(v, edge.IndexListExpr_Indices, n.Indices)

	case *ast.SliceExpr:
		walk(v, edge.SliceExpr_X, -1, n.X)
		if n.Low != nil {
			walk(v, edge.SliceExpr_Low, -1, n.Low)
		}
		if n.High != nil {
			walk(v, edge.SliceExpr_High, -1, n.High)
		}
		if n.Max != nil {
			walk(v, edge.SliceExpr_Max, -1, n.Max)
		}

	case *ast.TypeAssertExpr:
		walk(v, edge.TypeAssertExpr_X, -1, n.X)
		if n.Type != nil {
			walk(v, edge.TypeAssertExpr_Type, -1, n.Type)
		}

	case *ast.CallExpr:
		walk(v, edge.CallExpr_Fun, -1, n.Fun)
		walkList(v, edge.CallExpr_Args, n.Args)

	case *ast.StarExpr:
		walk(v, edge.StarExpr_X, -1, n.X)

	case *ast.UnaryExpr:
		walk(v, edge.UnaryExpr_X, -1, n.X)

	case *ast.BinaryExpr:
		walk(v, edge.BinaryExpr_X, -1, n.X)
		walk(v, edge.BinaryExpr_Y, -1, n.Y)

	case *ast.KeyValueExpr:
		walk(v, edge.KeyValueExpr_Key, -1, n.Key)
		walk(v, edge.KeyValueExpr_Value, -1, n.Value)

	// Types
	case *ast.ArrayType:
		if n.Len != nil {
			walk(v, edge.ArrayType_Len, -1, n.Len)
		}
		walk(v, edge.ArrayType_Elt, -1, n.Elt)

	case *ast.StructType:
		walk(v, edge.StructType_Fields, -1, n.Fields)

	case *ast.FuncType:
		if n.TypeParams != nil {
			walk(v, edge.FuncType_TypeParams, -1, n.TypeParams)
		}
		if n.Params != nil {
			walk(v, edge.FuncType_Params, -1, n.Params)
		}
		if n.Results != nil {
			walk(v, edge.FuncType_Results, -1, n.Results)
		}

	case *ast.InterfaceType:
		walk(v, edge.InterfaceType_Methods, -1, n.Methods)

	case *ast.MapType:
		walk(v, edge.MapType_Key, -1, n.Key)
		walk(v, edge.MapType_Value, -1, n.Value)

	case *ast.ChanType:
		walk(v, edge.ChanType_Value, -1, n.Value)

	// Statements
	case *ast.BadStmt:
		// nothing to do

	case *ast.DeclStmt:
		walk(v, edge.DeclStmt_Decl, -1, n.Decl)

	case *ast.EmptyStmt:
		// nothing to do

	case *ast.LabeledStmt:
		walk(v, edge.LabeledStmt_Label, -1, n.Label)
		walk(v, edge.LabeledStmt_Stmt, -1, n.Stmt)

	case *ast.ExprStmt:
		walk(v, edge.ExprStmt_X, -1, n.X)

	case *ast.SendStmt:
		walk(v, edge.SendStmt_Chan, -1, n.Chan)
		walk(v, edge.SendStmt_Value, -1, n.Value)

	case *ast.IncDecStmt:
		walk(v, edge.IncDecStmt_X, -1, n.X)

	case *ast.AssignStmt:
		walkList(v, edge.AssignStmt_Lhs, n.Lhs)
		walkList(v, edge.AssignStmt_Rhs, n.Rhs)

	case *ast.GoStmt:
		walk(v, edge.GoStmt_Call, -1, n.Call)

	case *ast.DeferStmt:
		walk(v, edge.DeferStmt_Call, -1, n.Call)

	case *ast.ReturnStmt:
		walkList(v, edge.ReturnStmt_Results, n.Results)

	case *ast.BranchStmt:
		if n.Label != nil {
			walk(v, edge.BranchStmt_Label, -1, n.Label)
		}

	case *ast.BlockStmt:
		walkList(v, edge.BlockStmt_List, n.List)

	case *ast.IfStmt:
		if n.Init != nil {
			walk(v, edge.IfStmt_Init, -1, n.Init)
		}
		walk(v, edge.IfStmt_Cond, -1, n.Cond)
		walk(v, edge.IfStmt_Body, -1, n.Body)
		if n.Else != nil {
			walk(v, edge.IfStmt_Else, -1, n.Else)
		}

	case *ast.CaseClause:
		walkList(v, edge.CaseClause_List, n.List)
		walkList(v, edge.CaseClause_Body, n.Body)

	case *ast.SwitchStmt:
		if n.Init != nil {
			walk(v, edge.SwitchStmt_Init, -1, n.Init)
		}
		if n.Tag != nil {
			walk(v, edge.SwitchStmt_Tag, -1, n.Tag)
		}
		walk(v, edge.SwitchStmt_Body, -1, n.Body)

	case *ast.TypeSwitchStmt:
		if n.Init != nil {
			walk(v, edge.TypeSwitchStmt_Init, -1, n.Init)
		}
		walk(v, edge.TypeSwitchStmt_Assign, -1, n.Assign)
		walk(v, edge.TypeSwitchStmt_Body, -1, n.Body)

	case *ast.CommClause:
		if n.Comm != nil {
			walk(v, edge.CommClause_Comm, -1, n.Comm)
		}
		walkList(v, edge.CommClause_Body, n.Body)

	case *ast.SelectStmt:
		walk(v, edge.SelectStmt_Body, -1, n.Body)

	case *ast.ForStmt:
		if n.Init != nil {
			walk(v, edge.ForStmt_Init, -1, n.Init)
		}
		if n.Cond != nil {
			walk(v, edge.ForStmt_Cond, -1, n.Cond)
		}
		if n.Post != nil {
			walk(v, edge.ForStmt_Post, -1, n.Post)
		}
		walk(v, edge.ForStmt_Body, -1, n.Body)

	case *ast.RangeStmt:
		if n.Key != nil {
			walk(v, edge.RangeStmt_Key, -1, n.Key)
		}
		if n.Value != nil {
			walk(v, edge.RangeStmt_Value, -1, n.Value)
		}
		walk(v, edge.RangeStmt_X, -1, n.X)
		walk(v, edge.RangeStmt_Body, -1, n.Body)

	// Declarations
	case *ast.ImportSpec:
		if n.Doc != nil {
			walk(v, edge.ImportSpec_Doc, -1, n.Doc)
		}
		if n.Name != nil {
			walk(v, edge.ImportSpec_Name, -1, n.Name)
		}
		walk(v, edge.ImportSpec_Path, -1, n.Path)
		if n.Comment != nil {
			walk(v, edge.ImportSpec_Comment, -1, n.Comment)
		}

	case *ast.ValueSpec:
		if n.Doc != nil {
			walk(v, edge.ValueSpec_Doc, -1, n.Doc)
		}
		walkList(v, edge.ValueSpec_Names, n.Names)
		if n.Type != nil {
			walk(v, edge.ValueSpec_Type, -1, n.Type)
		}
		walkList(v, edge.ValueSpec_Values, n.Values)
		if n.Comment != nil {
			walk(v, edge.ValueSpec_Comment, -1, n.Comment)
		}

	case *ast.TypeSpec:
		if n.Doc != nil {
			walk(v, edge.TypeSpec_Doc, -1, n.Doc)
		}
		walk(v, edge.TypeSpec_Name, -1, n.Name)
		if n.TypeParams != nil {
			walk(v, edge.TypeSpec_TypeParams, -1, n.TypeParams)
		}
		walk(v, edge.TypeSpec_Type, -1, n.Type)
		if n.Comment != nil {
			walk(v, edge.TypeSpec_Comment, -1, n.Comment)
		}

	case *ast.BadDecl:
		// nothing to do

	case *ast.GenDecl:
		if n.Doc != nil {
			walk(v, edge.GenDecl_Doc, -1, n.Doc)
		}
		walkList(v, edge.GenDecl_Specs, n.Specs)

	case *ast.FuncDecl:
		if n.Doc != nil {
			walk(v, edge.FuncDecl_Doc, -1, n.Doc)
		}
		if n.Recv != nil {
			walk(v, edge.FuncDecl_Recv, -1, n.Recv)
		}
		walk(v, edge.FuncDecl_Name, -1, n.Name)
		walk(v, edge.FuncDecl_Type, -1, n.Type)
		if n.Body != nil {
			walk(v, edge.FuncDecl_Body, -1, n.Body)
		}

	case *ast.File:
		if n.Doc != nil {
			walk(v, edge.File_Doc, -1, n.Doc)
		}
		walk(v, edge.File_Name, -1, n.Name)
		walkList(v, edge.File_Decls, n.Decls)
		// don't walk n.Comments - they have been
		// visited already through the individual
		// nodes

	default:
		// (includes *ast.Package)
		panic(fmt.Sprintf("Walk: unexpected node type %T", n))
	}

	v.pop(node)
}
