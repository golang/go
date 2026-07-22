// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modernize

import (
	"go/ast"
	"go/token"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/edge"
	"golang.org/x/tools/internal/analysis/analyzerutil"
	typeindexanalyzer "golang.org/x/tools/internal/analysis/typeindex"
	"golang.org/x/tools/internal/astutil"
	"golang.org/x/tools/internal/refactor"
	"golang.org/x/tools/internal/typesinternal"
	"golang.org/x/tools/internal/typesinternal/typeindex"
	"golang.org/x/tools/internal/versions"
)

var reflectTypeAssertAnalyzer = &analysis.Analyzer{
	Name: "reflecttypeassert",
	Doc:  analyzerutil.MustExtractDoc(doc, "reflecttypeassert"),
	Requires: []*analysis.Analyzer{
		inspect.Analyzer,
		typeindexanalyzer.Analyzer,
	},
	Run: reflecttypeassert,
	URL: "https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/modernize#hdr-Analyzer_reflecttypeassert",
}

func reflecttypeassert(pass *analysis.Pass) (any, error) {
	var (
		index = pass.ResultOf[typeindexanalyzer.Analyzer].(*typeindex.Index)
		info  = pass.TypesInfo

		valueInterface = index.Selection("reflect", "Value", "Interface")
	)

	for curCall := range index.Calls(valueInterface) {
		call := curCall.Node().(*ast.CallExpr)
		// Have: v.Interface()

		sel, ok := call.Fun.(*ast.SelectorExpr)
		if !ok {
			continue // method expression reflect.Value.Interface(v)
		}

		// TypeAssert's argument must be a reflect.Value; a pointer
		// receiver would need an explicit dereference in the rewrite.
		if !typesinternal.IsTypeNamed(info.TypeOf(sel.X), "reflect", "Value") {
			continue
		}

		// The call must be the operand of a type assertion
		// (not a type switch, whose Type field is nil).
		curOperand := astutil.UnparenEnclosingCursor(curCall)
		if curOperand.ParentEdgeKind() != edge.TypeAssertExpr_X {
			continue
		}
		curAssert := curOperand.Parent()
		assert := curAssert.Node().(*ast.TypeAssertExpr)
		if assert.Type == nil {
			continue // type switch
		}

		// The assertion must be the sole RHS of a two-valued
		// assignment, x, ok := v.Interface().(T), so that the
		// rewrite preserves the "commaOK" semantics; a single-valued
		// assertion panics on failure whereas TypeAssert does not.
		curRhs := astutil.UnparenEnclosingCursor(curAssert)
		if curRhs.ParentEdgeKind() != edge.AssignStmt_Rhs {
			continue
		}
		assign := curRhs.Parent().Node().(*ast.AssignStmt)
		if len(assign.Lhs) != 2 || len(assign.Rhs) != 1 ||
			(assign.Tok != token.ASSIGN && assign.Tok != token.DEFINE) {
			continue
		}

		file := astutil.EnclosingFile(curCall)
		if !analyzerutil.FileUsesGoVersion(pass, file, versions.Go1_25) {
			continue // TypeAssert requires go1.25
		}

		prefix, importEdits := refactor.AddImport(info, file, "reflect", "reflect", "TypeAssert", assert.Pos())

		tstr := astutil.Format(pass.Fset, assert.Type)
		pass.Report(analysis.Diagnostic{
			Pos:     assert.Pos(),
			End:     assert.End(),
			Message: "Interface().(" + tstr + ") can be simplified using reflect.TypeAssert",
			SuggestedFixes: []analysis.SuggestedFix{{
				// v.Interface().(T)  ->  reflect.TypeAssert[T](v)
				Message: "Replace Interface().(" + tstr + ") by reflect.TypeAssert[" + tstr + "]",
				// Edit around sel.X instead of reformatting it, so its
				// comments and spacing are preserved; only the type,
				// which must move, is reformatted.
				TextEdits: append(importEdits,
					analysis.TextEdit{
						Pos:     assert.Pos(),
						End:     sel.X.Pos(),
						NewText: []byte(prefix + "TypeAssert[" + tstr + "]("),
					},
					analysis.TextEdit{
						Pos:     sel.X.End(),
						End:     assert.End(),
						NewText: []byte(")"),
					},
				),
			}},
		})
	}

	return nil, nil
}
