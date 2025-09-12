// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package assign

// TODO(adonovan): check also for assignments to struct fields inside
// methods that are on T instead of *T.

import (
	_ "embed"
	"go/ast"
	"go/token"
	"go/types"
	"reflect"
	"strings"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/analysis/passes/internal/analysisutil"
	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/internal/analysisinternal"
)

//go:embed doc.go
var doc string

var Analyzer = &analysis.Analyzer{
	Name:     "assign",
	Doc:      analysisutil.MustExtractDoc(doc, "assign"),
	URL:      "https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/assign",
	Requires: []*analysis.Analyzer{inspect.Analyzer},
	Run:      run,
}

func run(pass *analysis.Pass) (any, error) {
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

		// Delete redundant LHS, RHS pairs, taking care
		// to include intervening commas.
		var (
			exprs                    []string // expressions appearing on both sides (x = x)
			edits                    []analysis.TextEdit
			runStartLHS, runStartRHS token.Pos // non-zero => within a run
		)
		for i, lhs := range stmt.Lhs {
			rhs := stmt.Rhs[i]
			isSelfAssign := false
			var le string

			if !analysisutil.HasSideEffects(pass.TypesInfo, lhs) &&
				!analysisutil.HasSideEffects(pass.TypesInfo, rhs) &&
				!isMapIndex(pass.TypesInfo, lhs) &&
				reflect.TypeOf(lhs) == reflect.TypeOf(rhs) { // short-circuit the heavy-weight gofmt check

				le = analysisinternal.Format(pass.Fset, lhs)
				re := analysisinternal.Format(pass.Fset, rhs)
				if le == re {
					isSelfAssign = true
				}
			}

			if isSelfAssign {
				exprs = append(exprs, le)
				if !runStartLHS.IsValid() {
					// Start of a new run of self-assignments.
					if i > 0 {
						runStartLHS = stmt.Lhs[i-1].End()
						runStartRHS = stmt.Rhs[i-1].End()
					} else {
						runStartLHS = lhs.Pos()
						runStartRHS = rhs.Pos()
					}
				}
			} else if runStartLHS.IsValid() {
				// End of a run of self-assignments.
				endLHS, endRHS := stmt.Lhs[i-1].End(), stmt.Rhs[i-1].End()
				if runStartLHS == stmt.Lhs[0].Pos() {
					endLHS, endRHS = lhs.Pos(), rhs.Pos()
				}
				edits = append(edits,
					analysis.TextEdit{Pos: runStartLHS, End: endLHS},
					analysis.TextEdit{Pos: runStartRHS, End: endRHS},
				)
				runStartLHS, runStartRHS = 0, 0
			}
		}

		// If a run of self-assignments continues to the end of the statement, close it.
		if runStartLHS.IsValid() {
			last := len(stmt.Lhs) - 1
			edits = append(edits,
				analysis.TextEdit{Pos: runStartLHS, End: stmt.Lhs[last].End()},
				analysis.TextEdit{Pos: runStartRHS, End: stmt.Rhs[last].End()},
			)
		}

		if len(exprs) == 0 {
			return
		}

		if len(exprs) == len(stmt.Lhs) {
			// If every part of the statement is a self-assignment,
			// remove the whole statement.
			edits = []analysis.TextEdit{{Pos: stmt.Pos(), End: stmt.End()}}
		}

		pass.Report(analysis.Diagnostic{
			Pos:     stmt.Pos(),
			Message: "self-assignment of " + strings.Join(exprs, ", "),
			SuggestedFixes: []analysis.SuggestedFix{{
				Message:   "Remove self-assignment",
				TextEdits: edits,
			}},
		})
	})

	return nil, nil
}

// isMapIndex returns true if e is a map index expression.
func isMapIndex(info *types.Info, e ast.Expr) bool {
	if idx, ok := ast.Unparen(e).(*ast.IndexExpr); ok {
		if typ := info.Types[idx.X].Type; typ != nil {
			_, ok := typ.Underlying().(*types.Map)
			return ok
		}
	}
	return false
}
