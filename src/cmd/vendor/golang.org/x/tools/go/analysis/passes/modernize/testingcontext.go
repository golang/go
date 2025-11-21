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
	"unicode"
	"unicode/utf8"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/edge"
	"golang.org/x/tools/go/types/typeutil"
	"golang.org/x/tools/internal/analysis/analyzerutil"
	typeindexanalyzer "golang.org/x/tools/internal/analysis/typeindex"
	"golang.org/x/tools/internal/astutil"
	"golang.org/x/tools/internal/typesinternal"
	"golang.org/x/tools/internal/typesinternal/typeindex"
	"golang.org/x/tools/internal/versions"
)

var TestingContextAnalyzer = &analysis.Analyzer{
	Name: "testingcontext",
	Doc:  analyzerutil.MustExtractDoc(doc, "testingcontext"),
	Requires: []*analysis.Analyzer{
		inspect.Analyzer,
		typeindexanalyzer.Analyzer,
	},
	Run: testingContext,
	URL: "https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/modernize#testingcontext",
}

// The testingContext pass replaces calls to context.WithCancel from within
// tests to a use of testing.{T,B,F}.Context(), added in Go 1.24.
//
// Specifically, the testingContext pass suggests to replace:
//
//	ctx, cancel := context.WithCancel(context.Background()) // or context.TODO
//	defer cancel()
//
// with:
//
//	ctx := t.Context()
//
// provided:
//
//   - ctx and cancel are declared by the assignment
//   - the deferred call is the only use of cancel
//   - the call is within a test or subtest function
//   - the relevant testing.{T,B,F} is named and not shadowed at the call
func testingContext(pass *analysis.Pass) (any, error) {
	var (
		index = pass.ResultOf[typeindexanalyzer.Analyzer].(*typeindex.Index)
		info  = pass.TypesInfo

		contextWithCancel = index.Object("context", "WithCancel")
	)

calls:
	for cur := range index.Calls(contextWithCancel) {
		call := cur.Node().(*ast.CallExpr)
		// Have: context.WithCancel(...)

		arg, ok := call.Args[0].(*ast.CallExpr)
		if !ok {
			continue
		}
		if !typesinternal.IsFunctionNamed(typeutil.Callee(info, arg), "context", "Background", "TODO") {
			continue
		}
		// Have: context.WithCancel(context.{Background,TODO}())

		parent := cur.Parent()
		assign, ok := parent.Node().(*ast.AssignStmt)
		if !ok || assign.Tok != token.DEFINE {
			continue
		}
		// Have: a, b := context.WithCancel(context.{Background,TODO}())

		// Check that both a and b are declared, not redeclarations.
		var lhs []types.Object
		for _, expr := range assign.Lhs {
			id, ok := expr.(*ast.Ident)
			if !ok {
				continue calls
			}
			obj, ok := info.Defs[id]
			if !ok {
				continue calls
			}
			lhs = append(lhs, obj)
		}

		next, ok := parent.NextSibling()
		if !ok {
			continue
		}
		defr, ok := next.Node().(*ast.DeferStmt)
		if !ok {
			continue
		}
		deferId, ok := defr.Call.Fun.(*ast.Ident)
		if !ok || !soleUseIs(index, lhs[1], deferId) {
			continue // b is used elsewhere
		}
		// Have:
		// a, b := context.WithCancel(context.{Background,TODO}())
		// defer b()

		// Check that we are in a test func.
		var testObj types.Object // relevant testing.{T,B,F}, or nil
		if curFunc, ok := enclosingFunc(cur); ok {
			switch n := curFunc.Node().(type) {
			case *ast.FuncLit:
				if ek, idx := curFunc.ParentEdge(); ek == edge.CallExpr_Args && idx == 1 {
					// Have: call(..., func(...) { ...context.WithCancel(...)... })
					obj := typeutil.Callee(info, curFunc.Parent().Node().(*ast.CallExpr))
					if (typesinternal.IsMethodNamed(obj, "testing", "T", "Run") ||
						typesinternal.IsMethodNamed(obj, "testing", "B", "Run")) &&
						len(n.Type.Params.List[0].Names) == 1 {

						// Have tb.Run(..., func(..., tb *testing.[TB]) { ...context.WithCancel(...)... }
						testObj = info.Defs[n.Type.Params.List[0].Names[0]]
					}
				}

			case *ast.FuncDecl:
				testObj = isTestFn(info, n)
			}
		}
		if testObj != nil && analyzerutil.FileUsesGoVersion(pass, astutil.EnclosingFile(cur), versions.Go1_24) {
			// Have a test function. Check that we can resolve the relevant
			// testing.{T,B,F} at the current position.
			if _, obj := lhs[0].Parent().LookupParent(testObj.Name(), lhs[0].Pos()); obj == testObj {
				pass.Report(analysis.Diagnostic{
					Pos:     call.Fun.Pos(),
					End:     call.Fun.End(),
					Message: fmt.Sprintf("context.WithCancel can be modernized using %s.Context", testObj.Name()),
					SuggestedFixes: []analysis.SuggestedFix{{
						Message: fmt.Sprintf("Replace context.WithCancel with %s.Context", testObj.Name()),
						TextEdits: []analysis.TextEdit{{
							Pos:     assign.Pos(),
							End:     defr.End(),
							NewText: fmt.Appendf(nil, "%s := %s.Context()", lhs[0].Name(), testObj.Name()),
						}},
					}},
				})
			}
		}
	}
	return nil, nil
}

// soleUseIs reports whether id is the sole Ident that uses obj.
// (It returns false if there were no uses of obj.)
func soleUseIs(index *typeindex.Index, obj types.Object, id *ast.Ident) bool {
	empty := true
	for use := range index.Uses(obj) {
		empty = false
		if use.Node() != id {
			return false
		}
	}
	return !empty
}

// isTestFn checks whether fn is a test function (TestX, BenchmarkX, FuzzX),
// returning the corresponding types.Object of the *testing.{T,B,F} argument.
// Returns nil if fn is a test function, but the testing.{T,B,F} argument is
// unnamed (or _).
//
// TODO(rfindley): consider handling the case of an unnamed argument, by adding
// an edit to give the argument a name.
//
// Adapted from go/analysis/passes/tests.
// TODO(rfindley): consider refactoring to share logic.
func isTestFn(info *types.Info, fn *ast.FuncDecl) types.Object {
	// Want functions with 0 results and 1 parameter.
	if fn.Type.Results != nil && len(fn.Type.Results.List) > 0 ||
		fn.Type.Params == nil ||
		len(fn.Type.Params.List) != 1 ||
		len(fn.Type.Params.List[0].Names) != 1 {

		return nil
	}

	prefix := testKind(fn.Name.Name)
	if prefix == "" {
		return nil
	}

	if tparams := fn.Type.TypeParams; tparams != nil && len(tparams.List) > 0 {
		return nil // test functions must not be generic
	}

	obj := info.Defs[fn.Type.Params.List[0].Names[0]]
	if obj == nil {
		return nil // e.g. _ *testing.T
	}

	var name string
	switch prefix {
	case "Test":
		name = "T"
	case "Benchmark":
		name = "B"
	case "Fuzz":
		name = "F"
	}

	if !typesinternal.IsPointerToNamed(obj.Type(), "testing", name) {
		return nil
	}
	return obj
}

// testKind returns "Test", "Benchmark", or "Fuzz" if name is a valid resp.
// test, benchmark, or fuzz function name. Otherwise, isTestName returns "".
//
// Adapted from go/analysis/passes/tests.isTestName.
func testKind(name string) string {
	var prefix string
	switch {
	case strings.HasPrefix(name, "Test"):
		prefix = "Test"
	case strings.HasPrefix(name, "Benchmark"):
		prefix = "Benchmark"
	case strings.HasPrefix(name, "Fuzz"):
		prefix = "Fuzz"
	}
	if prefix == "" {
		return ""
	}
	suffix := name[len(prefix):]
	if len(suffix) == 0 {
		// "Test" is ok.
		return prefix
	}
	r, _ := utf8.DecodeRuneInString(suffix)
	if unicode.IsLower(r) {
		return ""
	}
	return prefix
}
