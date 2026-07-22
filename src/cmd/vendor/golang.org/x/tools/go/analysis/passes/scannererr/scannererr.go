// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package scannererr defines an analyzer for uses of bufio.Scanner
// in which the user has forgotten to check Scanner.Err.
package scannererr

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/types"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/edge"
	typeindexanalyzer "golang.org/x/tools/internal/analysis/typeindex"
	"golang.org/x/tools/internal/moreiters"
	"golang.org/x/tools/internal/typesinternal"
	"golang.org/x/tools/internal/typesinternal/typeindex"
)

const doc = `scannererr: report failure to check bufio.Scanner.Err

This analyzer reports uses of bufio.Scanner in which the result of
NewScanner is assigned to a local variable that is then used in a loop
that calls Scanner.Scan, but lacks a final check of Scanner.Err,
which is how I/O errors are reported.

For example:

	sc := bufio.NewScanner(os.Stdin) // error: "bufio.Scanner sc is used in Scan loop without final check of sc.Err()"
	for sc.Scan() {
		line := sc.Text()
		use(line)
	}
	/* ...no use of sc.Err()... */

To avoid false positives, the analyzer is silent if the scanner is
passed into or out of the function or assigned somewhere other than a
local variable.

It is not this analyzer's goal to ensure proper handling of errors in
all cases, but merely the simple mistakes where the user may have been
oblivious to the existence of the Scanner.Err method.

The analyzer ignores calls to bufio.NewScanner whose argument is an
infallible memory-backed io.Reader such as strings.Reader or bytes.Buffer.
(In such cases, Scan may yet fail if a token or line is too long for the
scanner's internal buffer, but this is rare.)

If you know that errors are impossible for a given scanner, you can
suppress the diagnostic thus:

	_ = sc.Err() // ignore error; neither reading nor scanning can fail

`

var Analyzer = &analysis.Analyzer{
	Name:     "scannererr",
	Doc:      doc,
	URL:      "https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/scannererr",
	Requires: []*analysis.Analyzer{inspect.Analyzer, typeindexanalyzer.Analyzer},
	Run:      run,
}

func run(pass *analysis.Pass) (any, error) {
	var (
		index = pass.ResultOf[typeindexanalyzer.Analyzer].(*typeindex.Index)
		info  = pass.TypesInfo
	)

callLoop:
	for curCall := range index.Calls(index.Object("bufio", "NewScanner")) {
		// If NewScanner's reader operand is infallible (such
		// as strings.Reader, bytes.Reader, or bytes.Buffer),
		// disregard it. (Scanning can still fail if a token
		// or line is longer than the internal buffer, but
		// that's much less common than an I/O error).
		tArg := info.TypeOf(curCall.Node().(*ast.CallExpr).Args[0])
		if typesinternal.IsPointerToNamed(tArg, "bytes", "Buffer", "Reader") ||
			typesinternal.IsPointerToNamed(tArg, "strings", "Reader") {
			continue
		}

		// TODO(adonovan): factor this common pattern.
		var lhs ast.Expr
		switch ek, idx := curCall.ParentEdge(); ek {
		case edge.ValueSpec_Values:
			// var sc = bufio.NewScanner(...)
			curName := curCall.Parent().ChildAt(edge.ValueSpec_Names, idx)
			lhs = curName.Node().(*ast.Ident)
		case edge.AssignStmt_Rhs:
			// sc := bufio.NewScanner(...)   (or '=')
			curLhs := curCall.Parent().ChildAt(edge.AssignStmt_Lhs, idx)
			lhs = curLhs.Node().(ast.Expr)
		}
		id, ok := lhs.(*ast.Ident)
		if !ok {
			continue
		}
		sc, ok := info.ObjectOf(id).(*types.Var)
		if !ok {
			continue
		}
		// Have: sc := bufio.NewScanner(...)

		// Check all uses of the var sc.
		scanLoop := token.NoPos // position of sc.Scan() call within a loop
		for curUse := range index.Uses(sc) {
			// If the var sc is used in a context other than sc.Method(...),
			// assume conservatively that it may escape, and reject this candidate.
			if curUse.ParentEdgeKind() != edge.SelectorExpr_X ||
				curUse.Parent().ParentEdgeKind() != edge.CallExpr_Fun {
				continue callLoop
			}

			switch curUse.Parent().Node().(*ast.SelectorExpr).Sel.Name {
			case "Err":
				// If the sc.Err method is called anywhere, reject this candidate.
				continue callLoop
			case "Scan":
				// The Scan call must be in a loop that intervenes the declaration of sc.
				if curLoop, ok := moreiters.First(curUse.Enclosing((*ast.RangeStmt)(nil), (*ast.ForStmt)(nil))); ok {
					if curLoop.Node().Pos() > sc.Pos() {
						scanLoop = curUse.Node().Pos()
					}
				}
			}
		}
		if !scanLoop.IsValid() {
			continue
		}

		pass.Report(analysis.Diagnostic{
			Pos: curCall.Node().Pos(),
			End: curCall.Node().End(),
			Message: fmt.Sprintf("bufio.Scanner %q is used in Scan loop at line %d without final check of %s.Err()",
				sc.Name(), pass.Fset.Position(scanLoop).Line, sc.Name()),
		})
	}

	return nil, nil
}
