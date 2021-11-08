// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package assign defines an Analyzer that detects useless assignments.
package assign

// TODO(adonovan): check also for assignments to struct fields inside
// methods that are on T instead of *T.

import (
	"fmt"
	"go/ast"
	"go/token"
	"reflect"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/analysis/passes/internal/analysisutil"
	"golang.org/x/tools/go/ast/inspector"
)

const Doc = `check for useless assignments

This checker reports assignments of the form x = x or a[i] = a[i].
These are almost always useless, and even when they aren't they are
usually a mistake.`

var Analyzer = &analysis.Analyzer{
	Name:     "assign",
	Doc:      Doc,
	Requires: []*analysis.Analyzer{inspect.Analyzer},
	Run:      run,
}

func run(pass *analysis.Pass) (interface{}, error) {
	inspect := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)

	nodeFilter := []ast.Node{
		(*ast.AssignStmt)(nil),
	}
	inspect.Preorder(nodeFilter, func(n ast.Node) {
		stmt := n.(*ast.AssignStmt)
		if stmt.Tok != token.ASSIGN {
			return // ignore :=
		}
		if len(stmt.Lhs) != len(stmt.Rhs) {
			// If LHS and RHS have different cardinality, they can't be the same.
			return
		}
		for i, lhs := range stmt.Lhs {
			rhs := stmt.Rhs[i]
			if analysisutil.HasSideEffects(pass.TypesInfo, lhs) ||
				analysisutil.HasSideEffects(pass.TypesInfo, rhs) {
				continue // expressions may not be equal
			}
			if reflect.TypeOf(lhs) != reflect.TypeOf(rhs) {
				continue // short-circuit the heavy-weight gofmt check
			}
			le := analysisutil.Format(pass.Fset, lhs)
			re := analysisutil.Format(pass.Fset, rhs)
			if le == re {
				pass.Report(analysis.Diagnostic{
					Pos: stmt.Pos(), Message: fmt.Sprintf("self-assignment of %s to %s", re, le),
					SuggestedFixes: []analysis.SuggestedFix{
						{Message: "Remove", TextEdits: []analysis.TextEdit{
							{Pos: stmt.Pos(), End: stmt.End(), NewText: []byte{}},
						}},
					},
				})
			}
		}
	})

	return nil, nil
}
