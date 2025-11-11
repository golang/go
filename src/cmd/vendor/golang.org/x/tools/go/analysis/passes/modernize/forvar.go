// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modernize

import (
	"go/ast"
	"go/token"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/internal/analysis/analyzerutil"
	"golang.org/x/tools/internal/astutil"
	"golang.org/x/tools/internal/refactor"
	"golang.org/x/tools/internal/versions"
)

var ForVarAnalyzer = &analysis.Analyzer{
	Name:     "forvar",
	Doc:      analyzerutil.MustExtractDoc(doc, "forvar"),
	Requires: []*analysis.Analyzer{inspect.Analyzer},
	Run:      forvar,
	URL:      "https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/modernize#forvar",
}

// forvar offers to fix unnecessary copying of a for variable
//
//	for _, x := range foo {
//		x := x // offer to remove this superfluous assignment
//	}
//
// Prerequisites:
// First statement in a range loop has to be <ident> := <ident>
// where the two idents are the same,
// and the ident is defined (:=) as a variable in the for statement.
// (Note that this 'fix' does not work for three clause loops
// because the Go specfilesUsingGoVersionsays "The variable used by each subsequent iteration
// is declared implicitly before executing the post statement and initialized to the
// value of the previous iteration's variable at that moment.")
//
// Variant: same thing in an IfStmt.Init, when the IfStmt is the sole
// loop body statement:
//
//	for _, x := range foo {
//		if x := x; cond { ... }
//	}
//
// (The restriction is necessary to avoid potential problems arising
// from merging two distinct variables.)
//
// This analyzer is synergistic with stditerators,
// which may create redundant "x := x" statements.
func forvar(pass *analysis.Pass) (any, error) {
	for curFile := range filesUsingGoVersion(pass, versions.Go1_22) {
		for curLoop := range curFile.Preorder((*ast.RangeStmt)(nil)) {
			loop := curLoop.Node().(*ast.RangeStmt)
			if loop.Tok != token.DEFINE {
				continue
			}
			isLoopVarRedecl := func(stmt ast.Stmt) bool {
				if assign, ok := stmt.(*ast.AssignStmt); ok &&
					assign.Tok == token.DEFINE &&
					len(assign.Lhs) == len(assign.Rhs) {

					for i, lhs := range assign.Lhs {
						if !(astutil.EqualSyntax(lhs, assign.Rhs[i]) &&
							(astutil.EqualSyntax(lhs, loop.Key) ||
								astutil.EqualSyntax(lhs, loop.Value))) {
							return false
						}
					}
					return true
				}
				return false
			}
			// Have: for k, v := range x { stmts }
			//
			// Delete the prefix of stmts that are
			// of the form k := k; v := v; k, v := k, v; v, k := v, k.
			for _, stmt := range loop.Body.List {
				if isLoopVarRedecl(stmt) {
					// { x := x; ... }
					//   ------
				} else if ifstmt, ok := stmt.(*ast.IfStmt); ok &&
					ifstmt.Init != nil &&
					len(loop.Body.List) == 1 && // must be sole statement in loop body
					isLoopVarRedecl(ifstmt.Init) {
					// if x := x; cond {
					//    ------
					stmt = ifstmt.Init
				} else {
					break // stop at first other statement
				}

				curStmt, _ := curLoop.FindNode(stmt)
				edits := refactor.DeleteStmt(pass.Fset.File(stmt.Pos()), curStmt)
				if len(edits) > 0 {
					pass.Report(analysis.Diagnostic{
						Pos:     stmt.Pos(),
						End:     stmt.End(),
						Message: "copying variable is unneeded",
						SuggestedFixes: []analysis.SuggestedFix{{
							Message:   "Remove unneeded redeclaration",
							TextEdits: edits,
						}},
					})
				}
			}
		}
	}
	return nil, nil
}
