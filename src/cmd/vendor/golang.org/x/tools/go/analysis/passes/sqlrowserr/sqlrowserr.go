// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package sqlrowserr defines an analyzer for uses of sql.Rows
// in which the user has forgotten to check Rows.Err.
package sqlrowserr

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/types"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/edge"
	"golang.org/x/tools/go/ast/inspector"
	typeindexanalyzer "golang.org/x/tools/internal/analysis/typeindex"
	"golang.org/x/tools/internal/moreiters"
	"golang.org/x/tools/internal/typesinternal/typeindex"
)

const doc = `sqlrowserr: report failure to check sql.Rows.Err

This analyzer reports uses of sql.Rows in which the result of a query
such as db.Query() is assigned to a local variable that is then used
in a loop that calls Rows.Next, but lacks a final check of Rows.Err.
This causes row iteration errors to be discarded.

For example:

	rows, err := db.Query("select ...") // error: "sql.Rows rows is used in Next loop without final check of rows.Err()"
	if err != nil {
		return err
	}
	defer rows.Close() // ignore error
	for rows.Next() {
		var x int
		if err := rows.Scan(&x); err != nil {
			return err
		}
		use(x)
	}
	/* ...no use of rows.Err()... */

Correct usage of sql.Rows demands both a call to Rows.Close to release
resources and a call to Rows.Err to report iteration errors. It is
not critical to report resource cleanup errors, but it is crucial to
report iteration errors as they would otherwise be indistinguishable
from a smaller result.

To avoid false positives, the analyzer is silent if the Rows is passed
into or out of the function or assigned somewhere other than a local
variable.

It is not this analyzer's goal to ensure proper handling of errors in
all cases, but merely the simple mistakes where the user may have been
oblivious to the existence of the Rows.Err method.
`

var Analyzer = &analysis.Analyzer{
	Name:     "sqlrowserr",
	Doc:      doc,
	URL:      "https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/sqlrowserr",
	Requires: []*analysis.Analyzer{inspect.Analyzer, typeindexanalyzer.Analyzer},
	Run:      run,
}

// TODO(adonovan): factor common structures with the scannererr analyzer.

func run(pass *analysis.Pass) (any, error) {
	var (
		index = pass.ResultOf[typeindexanalyzer.Analyzer].(*typeindex.Index)
		info  = pass.TypesInfo
	)

	checkCall := func(curCall inspector.Cursor) {
		var lhs ast.Expr
		switch curCall.ParentEdgeKind() {
		case edge.ValueSpec_Values:
			// var sc, err = db.Query(...)
			curName := curCall.Parent().ChildAt(edge.ValueSpec_Names, 0)
			lhs = curName.Node().(*ast.Ident)
		case edge.AssignStmt_Rhs:
			// sc, err := db.Query(...)   (or '=')
			curLhs := curCall.Parent().ChildAt(edge.AssignStmt_Lhs, 0)
			lhs = curLhs.Node().(ast.Expr)
		}
		id, ok := lhs.(*ast.Ident)
		if !ok {
			return
		}
		rows, ok := info.ObjectOf(id).(*types.Var)
		if !ok {
			return
		}
		// Have: rows, err := db.Query(...)

		// Check all uses of the var rows.
		nextLoop := token.NoPos // position of rows.Next() call within a loop
		for curUse := range index.Uses(rows) {
			// If the var rows is used in a context other than rows.Method(...),
			// assume conservatively that it may escape, and reject this candidate.
			if curUse.ParentEdgeKind() != edge.SelectorExpr_X ||
				curUse.Parent().ParentEdgeKind() != edge.CallExpr_Fun {
				return
			}

			switch curUse.Parent().Node().(*ast.SelectorExpr).Sel.Name {
			case "Err":
				// If the rows.Err method is called anywhere, reject this candidate.
				return
			case "Next":
				// The Next call must be in a loop that intervenes the declaration of rows.
				if curLoop, ok := moreiters.First(curUse.Enclosing((*ast.RangeStmt)(nil), (*ast.ForStmt)(nil))); ok {
					if curLoop.Node().Pos() > rows.Pos() {
						nextLoop = curUse.Node().Pos()
					}
				}
			}
		}
		if !nextLoop.IsValid() {
			return
		}
		pass.Report(analysis.Diagnostic{
			Pos: curCall.Node().Pos(),
			End: curCall.Node().End(),
			Message: fmt.Sprintf("sql.Rows %q is used in Next loop at line %d without final check of %s.Err()",
				rows.Name(), pass.Fset.Position(nextLoop).Line, rows.Name()),
		})
	}

	// Check each query method in the sql package that returns (*Rows, error).
	// (This could be generalized for arbitrary such functions...)
	for _, m := range [...][2]string{
		{"Conn", "QueryContext"},
		{"DB", "QueryContext"},
		{"DB", "Query"},
		{"Stmt", "QueryContext"},
		{"Stmt", "Query"},
		{"Tx", "QueryContext"},
		{"Tx", "Query"},
	} {
		for cur := range index.Calls(index.Selection("database/sql", m[0], m[1])) {
			checkCall(cur)
		}
	}

	return nil, nil
}
