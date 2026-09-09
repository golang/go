// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modernize

import (
	"fmt"
	"go/ast"
	"go/types"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/types/typeutil"
	"golang.org/x/tools/internal/analysis/analyzerutil"
	"golang.org/x/tools/internal/astutil"
	"golang.org/x/tools/internal/refactor"
	"golang.org/x/tools/internal/typesinternal"
	"golang.org/x/tools/internal/versions"
)

var slicesClipAnalyzer = &analysis.Analyzer{
	Name: "slicesclip",
	Doc:  analyzerutil.MustExtractDoc(doc, "slicesclip"),
	Requires: []*analysis.Analyzer{
		inspect.Analyzer,
	},
	Run: slicesclip,
	URL: "https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/modernize#hdr-Analyzer_slicesclip",
}

func slicesclip(pass *analysis.Pass) (any, error) {
	if within(pass, "slices", "runtime") {
		return nil, nil
	}
	info := pass.TypesInfo

	// isLenX reports whether e is a call len(x) where x is
	// syntactically identical to the operand x of the slice expr.
	isLenX := func(e, x ast.Expr) bool {
		call, ok := e.(*ast.CallExpr)
		if !ok || len(call.Args) != 1 {
			return false
		}
		return typeutil.Callee(info, call) == builtinLen &&
			astutil.EqualSyntax(call.Args[0], x)
	}

	for curFile := range filesUsingGoVersion(pass, versions.Go1_21) {
		file := curFile.Node().(*ast.File)

		for curSlice := range curFile.Preorder((*ast.SliceExpr)(nil)) {
			slice := curSlice.Node().(*ast.SliceExpr)
			_, ok := info.TypeOf(slice.X).Underlying().(*types.Slice) // in case x is an array/pointer to array
			if !slice.Slice3 || slice.Low != nil || !ok {
				continue
			}

			if isLenX(slice.High, slice.X) && isLenX(slice.Max, slice.X) && typesinternal.NoEffects(info, slice.X) {
				// Have x[:len(x):len(x)] -> slices.Clip(x)
				prefix, edits := refactor.AddImport(info, file, "slices", "slices", "Clip", slice.Pos())
				sx := astutil.Format(pass.Fset, slice.X)
				pass.Report(analysis.Diagnostic{
					Pos:     slice.Pos(),
					End:     slice.End(),
					Message: "x[:len(x):len(x)] can be simplified using slices.Clip",
					SuggestedFixes: []analysis.SuggestedFix{{
						Message: fmt.Sprintf("Replace with slices.Clip(%s)", sx),
						TextEdits: append(edits, analysis.TextEdit{
							Pos:     slice.Pos(),
							End:     slice.End(),
							NewText: fmt.Appendf(nil, "%sClip(%s)", prefix, sx),
						}),
					}},
				})
			}
		}
	}

	return nil, nil
}
