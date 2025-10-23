// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modernize

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"strings"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/go/types/typeutil"
	"golang.org/x/tools/internal/analysisinternal"
	"golang.org/x/tools/internal/analysisinternal/generated"
	typeindexanalyzer "golang.org/x/tools/internal/analysisinternal/typeindex"
	"golang.org/x/tools/internal/astutil"
	"golang.org/x/tools/internal/moreiters"
	"golang.org/x/tools/internal/typesinternal"
	"golang.org/x/tools/internal/typesinternal/typeindex"
)

var BLoopAnalyzer = &analysis.Analyzer{
	Name: "bloop",
	Doc:  analysisinternal.MustExtractDoc(doc, "bloop"),
	Requires: []*analysis.Analyzer{
		generated.Analyzer,
		inspect.Analyzer,
		typeindexanalyzer.Analyzer,
	},
	Run: bloop,
	URL: "https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/modernize#bloop",
}

// bloop updates benchmarks that use "for range b.N", replacing it
// with go1.24's b.Loop() and eliminating any preceding
// b.{Start,Stop,Reset}Timer calls.
//
// Variants:
//
//	for i := 0; i < b.N; i++ {}  =>   for b.Loop() {}
//	for range b.N {}
func bloop(pass *analysis.Pass) (any, error) {
	skipGenerated(pass)

	if !typesinternal.Imports(pass.Pkg, "testing") {
		return nil, nil
	}

	var (
		inspect = pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)
		index   = pass.ResultOf[typeindexanalyzer.Analyzer].(*typeindex.Index)
		info    = pass.TypesInfo
	)

	// edits computes the text edits for a matched for/range loop
	// at the specified cursor. b is the *testing.B value, and
	// (start, end) is the portion using b.N to delete.
	edits := func(curLoop inspector.Cursor, b ast.Expr, start, end token.Pos) (edits []analysis.TextEdit) {
		curFn, _ := enclosingFunc(curLoop)
		// Within the same function, delete all calls to
		// b.{Start,Stop,Timer} that precede the loop.
		filter := []ast.Node{(*ast.ExprStmt)(nil), (*ast.FuncLit)(nil)}
		curFn.Inspect(filter, func(cur inspector.Cursor) (descend bool) {
			node := cur.Node()
			if is[*ast.FuncLit](node) {
				return false // don't descend into FuncLits (e.g. sub-benchmarks)
			}
			stmt := node.(*ast.ExprStmt)
			if stmt.Pos() > start {
				return false // not preceding: stop
			}
			if call, ok := stmt.X.(*ast.CallExpr); ok {
				obj := typeutil.Callee(info, call)
				if typesinternal.IsMethodNamed(obj, "testing", "B", "StopTimer", "StartTimer", "ResetTimer") {
					// Delete call statement.
					// TODO(adonovan): delete following newline, or
					// up to start of next stmt? (May delete a comment.)
					edits = append(edits, analysis.TextEdit{
						Pos: stmt.Pos(),
						End: stmt.End(),
					})
				}
			}
			return true
		})

		// Replace ...b.N... with b.Loop().
		return append(edits, analysis.TextEdit{
			Pos:     start,
			End:     end,
			NewText: fmt.Appendf(nil, "%s.Loop()", astutil.Format(pass.Fset, b)),
		})
	}

	// Find all for/range statements.
	loops := []ast.Node{
		(*ast.ForStmt)(nil),
		(*ast.RangeStmt)(nil),
	}
	for curFile := range filesUsing(inspect, info, "go1.24") {
		for curLoop := range curFile.Preorder(loops...) {
			switch n := curLoop.Node().(type) {
			case *ast.ForStmt:
				// for _; i < b.N; _ {}
				if cmp, ok := n.Cond.(*ast.BinaryExpr); ok && cmp.Op == token.LSS {
					if sel, ok := cmp.Y.(*ast.SelectorExpr); ok &&
						sel.Sel.Name == "N" &&
						typesinternal.IsPointerToNamed(info.TypeOf(sel.X), "testing", "B") && usesBenchmarkNOnce(curLoop, info) {

						delStart, delEnd := n.Cond.Pos(), n.Cond.End()

						// Eliminate variable i if no longer needed:
						//  for i := 0; i < b.N; i++ {
						//    ...no references to i...
						//  }
						body, _ := curLoop.LastChild()
						if v := isIncrementLoop(info, n); v != nil &&
							!uses(index, body, v) {
							delStart, delEnd = n.Init.Pos(), n.Post.End()
						}

						pass.Report(analysis.Diagnostic{
							// Highlight "i < b.N".
							Pos:     n.Cond.Pos(),
							End:     n.Cond.End(),
							Message: "b.N can be modernized using b.Loop()",
							SuggestedFixes: []analysis.SuggestedFix{{
								Message:   "Replace b.N with b.Loop()",
								TextEdits: edits(curLoop, sel.X, delStart, delEnd),
							}},
						})
					}
				}

			case *ast.RangeStmt:
				// for range b.N {} -> for b.Loop() {}
				//
				// TODO(adonovan): handle "for i := range b.N".
				if sel, ok := n.X.(*ast.SelectorExpr); ok &&
					n.Key == nil &&
					n.Value == nil &&
					sel.Sel.Name == "N" &&
					typesinternal.IsPointerToNamed(info.TypeOf(sel.X), "testing", "B") && usesBenchmarkNOnce(curLoop, info) {

					pass.Report(analysis.Diagnostic{
						// Highlight "range b.N".
						Pos:     n.Range,
						End:     n.X.End(),
						Message: "b.N can be modernized using b.Loop()",
						SuggestedFixes: []analysis.SuggestedFix{{
							Message:   "Replace b.N with b.Loop()",
							TextEdits: edits(curLoop, sel.X, n.Range, n.X.End()),
						}},
					})
				}
			}
		}
	}
	return nil, nil
}

// uses reports whether the subtree cur contains a use of obj.
func uses(index *typeindex.Index, cur inspector.Cursor, obj types.Object) bool {
	for use := range index.Uses(obj) {
		if cur.Contains(use) {
			return true
		}
	}
	return false
}

// enclosingFunc returns the cursor for the innermost Func{Decl,Lit}
// that encloses c, if any.
func enclosingFunc(c inspector.Cursor) (inspector.Cursor, bool) {
	return moreiters.First(c.Enclosing((*ast.FuncDecl)(nil), (*ast.FuncLit)(nil)))
}

// usesBenchmarkNOnce reports whether a b.N loop should be modernized to b.Loop().
// Only modernize loops that are:
// 1. Directly in a benchmark function (not in nested functions)
//   - b.Loop() must be called in the same goroutine as the benchmark function
//   - Function literals are often used with goroutines (go func(){...})
//
// 2. The only b.N loop in that benchmark function
//   - b.Loop() can only be called once per benchmark execution
//   - Multiple calls result in "B.Loop called with timer stopped" error
func usesBenchmarkNOnce(c inspector.Cursor, info *types.Info) bool {
	// Find the enclosing benchmark function
	curFunc, ok := enclosingFunc(c)
	if !ok {
		return false
	}

	// Check if this is actually a benchmark function
	fdecl, ok := curFunc.Node().(*ast.FuncDecl)
	if !ok {
		return false // not in a function; or, inside a FuncLit
	}
	if !isBenchmarkFunc(fdecl) {
		return false
	}

	// Count b.N references in this benchmark function
	bnRefCount := 0
	filter := []ast.Node{(*ast.SelectorExpr)(nil), (*ast.FuncLit)(nil)}
	curFunc.Inspect(filter, func(cur inspector.Cursor) bool {
		switch n := cur.Node().(type) {
		case *ast.FuncLit:
			return false // don't descend into nested function literals
		case *ast.SelectorExpr:
			if n.Sel.Name == "N" && typesinternal.IsPointerToNamed(info.TypeOf(n.X), "testing", "B") {
				bnRefCount++
			}
		}
		return true
	})

	// Only modernize if there's exactly one b.N reference
	return bnRefCount == 1
}

// isBenchmarkFunc reports whether f is a benchmark function.
func isBenchmarkFunc(f *ast.FuncDecl) bool {
	return f.Recv == nil &&
		f.Name != nil &&
		f.Name.IsExported() &&
		strings.HasPrefix(f.Name.Name, "Benchmark") &&
		f.Type.Params != nil &&
		len(f.Type.Params.List) == 1
}

// isIncrementLoop reports whether loop has the form "for i := 0; ...; i++ { ... }",
// and if so, it returns the symbol for the index variable.
func isIncrementLoop(info *types.Info, loop *ast.ForStmt) *types.Var {
	if assign, ok := loop.Init.(*ast.AssignStmt); ok &&
		assign.Tok == token.DEFINE &&
		len(assign.Rhs) == 1 &&
		isZeroIntLiteral(info, assign.Rhs[0]) &&
		is[*ast.IncDecStmt](loop.Post) &&
		loop.Post.(*ast.IncDecStmt).Tok == token.INC &&
		astutil.EqualSyntax(loop.Post.(*ast.IncDecStmt).X, assign.Lhs[0]) {
		return info.Defs[assign.Lhs[0].(*ast.Ident)].(*types.Var)
	}
	return nil
}
