// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package inspector

// This file defines func typeOf(ast.Node) uint64.
//
// The initial map-based implementation was too slow;
// see https://go-review.googlesource.com/c/tools/+/135655/1/go/ast/inspector/inspector.go#196

import (
	"go/ast"
	"math"
)

const (
	nArrayType = iota
	nAssignStmt
	nBadDecl
	nBadExpr
	nBadStmt
	nBasicLit
	nBinaryExpr
	nBlockStmt
	nBranchStmt
	nCallExpr
	nCaseClause
	nChanType
	nCommClause
	nComment
	nCommentGroup
	nCompositeLit
	nDeclStmt
	nDeferStmt
	nEllipsis
	nEmptyStmt
	nExprStmt
	nField
	nFieldList
	nFile
	nForStmt
	nFuncDecl
	nFuncLit
	nFuncType
	nGenDecl
	nGoStmt
	nIdent
	nIfStmt
	nImportSpec
	nIncDecStmt
	nIndexExpr
	nIndexListExpr
	nInterfaceType
	nKeyValueExpr
	nLabeledStmt
	nMapType
	nPackage
	nParenExpr
	nRangeStmt
	nReturnStmt
	nSelectStmt
	nSelectorExpr
	nSendStmt
	nSliceExpr
	nStarExpr
	nStructType
	nSwitchStmt
	nTypeAssertExpr
	nTypeSpec
	nTypeSwitchStmt
	nUnaryExpr
	nValueSpec
)

// typeOf returns a distinct single-bit value that represents the type of n.
//
// Various implementations were benchmarked with BenchmarkNewInspector:
//
//	                                                                GOGC=off
//	- type switch					4.9-5.5ms	2.1ms
//	- binary search over a sorted list of types	5.5-5.9ms	2.5ms
//	- linear scan, frequency-ordered list		5.9-6.1ms	2.7ms
//	- linear scan, unordered list			6.4ms		2.7ms
//	- hash table					6.5ms		3.1ms
//
// A perfect hash seemed like overkill.
//
// The compiler's switch statement is the clear winner
// as it produces a binary tree in code,
// with constant conditions and good branch prediction.
// (Sadly it is the most verbose in source code.)
// Binary search suffered from poor branch prediction.
func typeOf(n ast.Node) uint64 {
	// Fast path: nearly half of all nodes are identifiers.
	if _, ok := n.(*ast.Ident); ok {
		return 1 << nIdent
	}

	// These cases include all nodes encountered by ast.Inspect.
	switch n.(type) {
	case *ast.ArrayType:
		return 1 << nArrayType
	case *ast.AssignStmt:
		return 1 << nAssignStmt
	case *ast.BadDecl:
		return 1 << nBadDecl
	case *ast.BadExpr:
		return 1 << nBadExpr
	case *ast.BadStmt:
		return 1 << nBadStmt
	case *ast.BasicLit:
		return 1 << nBasicLit
	case *ast.BinaryExpr:
		return 1 << nBinaryExpr
	case *ast.BlockStmt:
		return 1 << nBlockStmt
	case *ast.BranchStmt:
		return 1 << nBranchStmt
	case *ast.CallExpr:
		return 1 << nCallExpr
	case *ast.CaseClause:
		return 1 << nCaseClause
	case *ast.ChanType:
		return 1 << nChanType
	case *ast.CommClause:
		return 1 << nCommClause
	case *ast.Comment:
		return 1 << nComment
	case *ast.CommentGroup:
		return 1 << nCommentGroup
	case *ast.CompositeLit:
		return 1 << nCompositeLit
	case *ast.DeclStmt:
		return 1 << nDeclStmt
	case *ast.DeferStmt:
		return 1 << nDeferStmt
	case *ast.Ellipsis:
		return 1 << nEllipsis
	case *ast.EmptyStmt:
		return 1 << nEmptyStmt
	case *ast.ExprStmt:
		return 1 << nExprStmt
	case *ast.Field:
		return 1 << nField
	case *ast.FieldList:
		return 1 << nFieldList
	case *ast.File:
		return 1 << nFile
	case *ast.ForStmt:
		return 1 << nForStmt
	case *ast.FuncDecl:
		return 1 << nFuncDecl
	case *ast.FuncLit:
		return 1 << nFuncLit
	case *ast.FuncType:
		return 1 << nFuncType
	case *ast.GenDecl:
		return 1 << nGenDecl
	case *ast.GoStmt:
		return 1 << nGoStmt
	case *ast.Ident:
		return 1 << nIdent
	case *ast.IfStmt:
		return 1 << nIfStmt
	case *ast.ImportSpec:
		return 1 << nImportSpec
	case *ast.IncDecStmt:
		return 1 << nIncDecStmt
	case *ast.IndexExpr:
		return 1 << nIndexExpr
	case *ast.IndexListExpr:
		return 1 << nIndexListExpr
	case *ast.InterfaceType:
		return 1 << nInterfaceType
	case *ast.KeyValueExpr:
		return 1 << nKeyValueExpr
	case *ast.LabeledStmt:
		return 1 << nLabeledStmt
	case *ast.MapType:
		return 1 << nMapType
	case *ast.Package:
		return 1 << nPackage
	case *ast.ParenExpr:
		return 1 << nParenExpr
	case *ast.RangeStmt:
		return 1 << nRangeStmt
	case *ast.ReturnStmt:
		return 1 << nReturnStmt
	case *ast.SelectStmt:
		return 1 << nSelectStmt
	case *ast.SelectorExpr:
		return 1 << nSelectorExpr
	case *ast.SendStmt:
		return 1 << nSendStmt
	case *ast.SliceExpr:
		return 1 << nSliceExpr
	case *ast.StarExpr:
		return 1 << nStarExpr
	case *ast.StructType:
		return 1 << nStructType
	case *ast.SwitchStmt:
		return 1 << nSwitchStmt
	case *ast.TypeAssertExpr:
		return 1 << nTypeAssertExpr
	case *ast.TypeSpec:
		return 1 << nTypeSpec
	case *ast.TypeSwitchStmt:
		return 1 << nTypeSwitchStmt
	case *ast.UnaryExpr:
		return 1 << nUnaryExpr
	case *ast.ValueSpec:
		return 1 << nValueSpec
	}
	return 0
}

func maskOf(nodes []ast.Node) uint64 {
	if len(nodes) == 0 {
		return math.MaxUint64 // match all node types
	}
	var mask uint64
	for _, n := range nodes {
		mask |= typeOf(n)
	}
	return mask
}
