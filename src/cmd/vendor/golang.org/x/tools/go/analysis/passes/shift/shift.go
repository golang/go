// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package shift defines an Analyzer that checks for shifts that exceed
// the width of an integer.
package shift

// TODO(adonovan): integrate with ctrflow (CFG-based) dead code analysis. May
// have impedance mismatch due to its (non-)treatment of constant
// expressions (such as runtime.GOARCH=="386").

import (
	"go/ast"
	"go/constant"
	"go/token"
	"go/types"
	"math"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/analysis/passes/internal/analysisutil"
	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/internal/typeparams"
)

const Doc = "check for shifts that equal or exceed the width of the integer"

var Analyzer = &analysis.Analyzer{
	Name:     "shift",
	Doc:      Doc,
	URL:      "https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/shift",
	Requires: []*analysis.Analyzer{inspect.Analyzer},
	Run:      run,
}

func run(pass *analysis.Pass) (interface{}, error) {
	inspect := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)

	// Do a complete pass to compute dead nodes.
	dead := make(map[ast.Node]bool)
	nodeFilter := []ast.Node{
		(*ast.IfStmt)(nil),
		(*ast.SwitchStmt)(nil),
	}
	inspect.Preorder(nodeFilter, func(n ast.Node) {
		// TODO(adonovan): move updateDead into this file.
		updateDead(pass.TypesInfo, dead, n)
	})

	nodeFilter = []ast.Node{
		(*ast.AssignStmt)(nil),
		(*ast.BinaryExpr)(nil),
	}
	inspect.Preorder(nodeFilter, func(node ast.Node) {
		if dead[node] {
			// Skip shift checks on unreachable nodes.
			return
		}

		switch node := node.(type) {
		case *ast.BinaryExpr:
			if node.Op == token.SHL || node.Op == token.SHR {
				checkLongShift(pass, node, node.X, node.Y)
			}
		case *ast.AssignStmt:
			if len(node.Lhs) != 1 || len(node.Rhs) != 1 {
				return
			}
			if node.Tok == token.SHL_ASSIGN || node.Tok == token.SHR_ASSIGN {
				checkLongShift(pass, node, node.Lhs[0], node.Rhs[0])
			}
		}
	})
	return nil, nil
}

// checkLongShift checks if shift or shift-assign operations shift by more than
// the length of the underlying variable.
func checkLongShift(pass *analysis.Pass, node ast.Node, x, y ast.Expr) {
	if pass.TypesInfo.Types[x].Value != nil {
		// Ignore shifts of constants.
		// These are frequently used for bit-twiddling tricks
		// like ^uint(0) >> 63 for 32/64 bit detection and compatibility.
		return
	}

	v := pass.TypesInfo.Types[y].Value
	if v == nil {
		return
	}
	amt, ok := constant.Int64Val(v)
	if !ok {
		return
	}
	t := pass.TypesInfo.Types[x].Type
	if t == nil {
		return
	}
	var structuralTypes []types.Type
	switch t := t.(type) {
	case *types.TypeParam:
		terms, err := typeparams.StructuralTerms(t)
		if err != nil {
			return // invalid type
		}
		for _, term := range terms {
			structuralTypes = append(structuralTypes, term.Type())
		}
	default:
		structuralTypes = append(structuralTypes, t)
	}
	sizes := make(map[int64]struct{})
	for _, t := range structuralTypes {
		size := 8 * pass.TypesSizes.Sizeof(t)
		sizes[size] = struct{}{}
	}
	minSize := int64(math.MaxInt64)
	for size := range sizes {
		if size < minSize {
			minSize = size
		}
	}
	if amt >= minSize {
		ident := analysisutil.Format(pass.Fset, x)
		qualifier := ""
		if len(sizes) > 1 {
			qualifier = "may be "
		}
		pass.ReportRangef(node, "%s (%s%d bits) too small for shift of %d", ident, qualifier, minSize, amt)
	}
}
