// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modernize

import (
	"fmt"
	"go/ast"
	"go/types"
	"strings"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/edge"
	"golang.org/x/tools/internal/analysisinternal"
	"golang.org/x/tools/internal/analysisinternal/generated"
	typeindexanalyzer "golang.org/x/tools/internal/analysisinternal/typeindex"
	"golang.org/x/tools/internal/astutil"
	"golang.org/x/tools/internal/typesinternal/typeindex"
)

var FmtAppendfAnalyzer = &analysis.Analyzer{
	Name: "fmtappendf",
	Doc:  analysisinternal.MustExtractDoc(doc, "fmtappendf"),
	Requires: []*analysis.Analyzer{
		generated.Analyzer,
		inspect.Analyzer,
		typeindexanalyzer.Analyzer,
	},
	Run: fmtappendf,
	URL: "https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/modernize#fmtappendf",
}

// The fmtappend function replaces []byte(fmt.Sprintf(...)) by
// fmt.Appendf(nil, ...), and similarly for Sprint, Sprintln.
func fmtappendf(pass *analysis.Pass) (any, error) {
	skipGenerated(pass)

	index := pass.ResultOf[typeindexanalyzer.Analyzer].(*typeindex.Index)
	for _, fn := range []types.Object{
		index.Object("fmt", "Sprintf"),
		index.Object("fmt", "Sprintln"),
		index.Object("fmt", "Sprint"),
	} {
		for curCall := range index.Calls(fn) {
			call := curCall.Node().(*ast.CallExpr)
			if ek, idx := curCall.ParentEdge(); ek == edge.CallExpr_Args && idx == 0 {
				// Is parent a T(fmt.SprintX(...)) conversion?
				conv := curCall.Parent().Node().(*ast.CallExpr)
				tv := pass.TypesInfo.Types[conv.Fun]
				if tv.IsType() && types.Identical(tv.Type, byteSliceType) &&
					fileUses(pass.TypesInfo, astutil.EnclosingFile(curCall), "go1.19") {
					// Have: []byte(fmt.SprintX(...))

					// Find "Sprint" identifier.
					var id *ast.Ident
					switch e := ast.Unparen(call.Fun).(type) {
					case *ast.SelectorExpr:
						id = e.Sel // "fmt.Sprint"
					case *ast.Ident:
						id = e // "Sprint" after `import . "fmt"`
					}

					old, new := fn.Name(), strings.Replace(fn.Name(), "Sprint", "Append", 1)
					edits := []analysis.TextEdit{
						{
							// delete "[]byte("
							Pos: conv.Pos(),
							End: conv.Lparen + 1,
						},
						{
							// remove ")"
							Pos: conv.Rparen,
							End: conv.Rparen + 1,
						},
						{
							Pos:     id.Pos(),
							End:     id.End(),
							NewText: []byte(new),
						},
						{
							Pos:     call.Lparen + 1,
							NewText: []byte("nil, "),
						},
					}
					if len(conv.Args) == 1 {
						arg := conv.Args[0]
						// Determine if we have T(fmt.SprintX(...)<non-args,
						// like a space or a comma>). If so, delete the non-args
						// that come before the right parenthesis. Leaving an
						// extra comma here produces invalid code. (See
						// golang/go#74709)
						if arg.End() < conv.Rparen {
							edits = append(edits, analysis.TextEdit{
								Pos: arg.End(),
								End: conv.Rparen,
							})
						}
					}
					pass.Report(analysis.Diagnostic{
						Pos:     conv.Pos(),
						End:     conv.End(),
						Message: fmt.Sprintf("Replace []byte(fmt.%s...) with fmt.%s", old, new),
						SuggestedFixes: []analysis.SuggestedFix{{
							Message:   fmt.Sprintf("Replace []byte(fmt.%s...) with fmt.%s", old, new),
							TextEdits: edits,
						}},
					})
				}
			}
		}
	}
	return nil, nil
}
