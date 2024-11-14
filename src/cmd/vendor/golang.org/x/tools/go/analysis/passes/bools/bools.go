// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package bools defines an Analyzer that detects common mistakes
// involving boolean operators.
package bools

import (
	"go/ast"
	"go/token"
	"go/types"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/analysis/passes/internal/analysisutil"
	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/go/ast/inspector"
)

const Doc = "check for common mistakes involving boolean operators"

var Analyzer = &analysis.Analyzer{
	Name:     "bools",
	Doc:      Doc,
	URL:      "https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/bools",
	Requires: []*analysis.Analyzer{inspect.Analyzer},
	Run:      run,
}

func run(pass *analysis.Pass) (interface{}, error) {
	inspect := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)

	nodeFilter := []ast.Node{
		(*ast.BinaryExpr)(nil),
	}
	seen := make(map[*ast.BinaryExpr]bool)
	inspect.Preorder(nodeFilter, func { n ->
		e := n.(*ast.BinaryExpr)
		if seen[e] {
			// Already processed as a subexpression of an earlier node.
			return
		}

		var op boolOp
		switch e.Op {
		case token.LOR:
			op = or
		case token.LAND:
			op = and
		default:
			return
		}

		comm := op.commutativeSets(pass.TypesInfo, e, seen)
		for _, exprs := range comm {
			op.checkRedundant(pass, exprs)
			op.checkSuspect(pass, exprs)
		}
	})
	return nil, nil
}

type boolOp struct {
	name  string
	tok   token.Token // token corresponding to this operator
	badEq token.Token // token corresponding to the equality test that should not be used with this operator
}

var (
	or  = boolOp{"or", token.LOR, token.NEQ}
	and = boolOp{"and", token.LAND, token.EQL}
)

// commutativeSets returns all side effect free sets of
// expressions in e that are connected by op.
// For example, given 'a || b || f() || c || d' with the or op,
// commutativeSets returns {{b, a}, {d, c}}.
// commutativeSets adds any expanded BinaryExprs to seen.
func (op boolOp) commutativeSets(info *types.Info, e *ast.BinaryExpr, seen map[*ast.BinaryExpr]bool) [][]ast.Expr {
	exprs := op.split(e, seen)

	// Partition the slice of expressions into commutative sets.
	i := 0
	var sets [][]ast.Expr
	for j := 0; j <= len(exprs); j++ {
		if j == len(exprs) || analysisutil.HasSideEffects(info, exprs[j]) {
			if i < j {
				sets = append(sets, exprs[i:j])
			}
			i = j + 1
		}
	}

	return sets
}

// checkRedundant checks for expressions of the form
//
//	e && e
//	e || e
//
// Exprs must contain only side effect free expressions.
func (op boolOp) checkRedundant(pass *analysis.Pass, exprs []ast.Expr) {
	seen := make(map[string]bool)
	for _, e := range exprs {
		efmt := analysisutil.Format(pass.Fset, e)
		if seen[efmt] {
			pass.ReportRangef(e, "redundant %s: %s %s %s", op.name, efmt, op.tok, efmt)
		} else {
			seen[efmt] = true
		}
	}
}

// checkSuspect checks for expressions of the form
//
//	x != c1 || x != c2
//	x == c1 && x == c2
//
// where c1 and c2 are constant expressions.
// If c1 and c2 are the same then it's redundant;
// if c1 and c2 are different then it's always true or always false.
// Exprs must contain only side effect free expressions.
func (op boolOp) checkSuspect(pass *analysis.Pass, exprs []ast.Expr) {
	// seen maps from expressions 'x' to equality expressions 'x != c'.
	seen := make(map[string]string)

	for _, e := range exprs {
		bin, ok := e.(*ast.BinaryExpr)
		if !ok || bin.Op != op.badEq {
			continue
		}

		// In order to avoid false positives, restrict to cases
		// in which one of the operands is constant. We're then
		// interested in the other operand.
		// In the rare case in which both operands are constant
		// (e.g. runtime.GOOS and "windows"), we'll only catch
		// mistakes if the LHS is repeated, which is how most
		// code is written.
		var x ast.Expr
		switch {
		case pass.TypesInfo.Types[bin.Y].Value != nil:
			x = bin.X
		case pass.TypesInfo.Types[bin.X].Value != nil:
			x = bin.Y
		default:
			continue
		}

		// e is of the form 'x != c' or 'x == c'.
		xfmt := analysisutil.Format(pass.Fset, x)
		efmt := analysisutil.Format(pass.Fset, e)
		if prev, found := seen[xfmt]; found {
			// checkRedundant handles the case in which efmt == prev.
			if efmt != prev {
				pass.ReportRangef(e, "suspect %s: %s %s %s", op.name, efmt, op.tok, prev)
			}
		} else {
			seen[xfmt] = efmt
		}
	}
}

// split returns a slice of all subexpressions in e that are connected by op.
// For example, given 'a || (b || c) || d' with the or op,
// split returns []{d, c, b, a}.
// seen[e] is already true; any newly processed exprs are added to seen.
func (op boolOp) split(e ast.Expr, seen map[*ast.BinaryExpr]bool) (exprs []ast.Expr) {
	for {
		e = astutil.Unparen(e)
		if b, ok := e.(*ast.BinaryExpr); ok && b.Op == op.tok {
			seen[b] = true
			exprs = append(exprs, op.split(b.Y, seen)...)
			e = b.X
		} else {
			exprs = append(exprs, e)
			break
		}
	}
	return
}
