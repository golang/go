// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modernize

import (
	"fmt"
	"go/ast"
	"go/constant"
	"go/types"
	"strings"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/edge"
	"golang.org/x/tools/internal/analysis/analyzerutil"
	typeindexanalyzer "golang.org/x/tools/internal/analysis/typeindex"
	"golang.org/x/tools/internal/astutil"
	"golang.org/x/tools/internal/fmtstr"
	"golang.org/x/tools/internal/typesinternal/typeindex"
	"golang.org/x/tools/internal/versions"
)

var FmtAppendfAnalyzer = &analysis.Analyzer{
	Name: "fmtappendf",
	Doc:  analyzerutil.MustExtractDoc(doc, "fmtappendf"),
	Requires: []*analysis.Analyzer{
		inspect.Analyzer,
		typeindexanalyzer.Analyzer,
	},
	Run: fmtappendf,
	URL: "https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/modernize#fmtappendf",
}

// The fmtappend function replaces []byte(fmt.Sprintf(...)) by
// fmt.Appendf(nil, ...), and similarly for Sprint, Sprintln.
func fmtappendf(pass *analysis.Pass) (any, error) {
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
				info := pass.TypesInfo
				tv := info.Types[conv.Fun]
				if tv.IsType() && types.Identical(tv.Type, byteSliceType) {
					// Have: []byte(fmt.SprintX(...))
					if len(call.Args) == 0 {
						continue
					}
					// fmt.Sprint(f) and fmt.Append(f) have different nil semantics
					// when the format produces an empty string:
					// []byte(fmt.Sprintf("")) returns an empty but non-nil
					// []byte{}, while fmt.Appendf(nil, "") returns nil) so we
					// should skip these cases.
					if fn.Name() == "Sprint" || fn.Name() == "Sprintf" {
						format := info.Types[call.Args[0]].Value
						if format != nil && mayFormatEmpty(constant.StringVal(format)) {
							continue
						}
					}

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
							// Delete "[]byte(", including any spaces before the first argument.
							Pos: conv.Pos(),
							End: conv.Args[0].Pos(), // always exactly one argument in a valid byte slice conversion
						},
						{
							// Delete ")", including any non-args (space or
							// commas) that come before the right parenthesis.
							// Leaving an extra comma here produces invalid
							// code. (See golang/go#74709)
							// Unfortunately, this and the edit above may result
							// in deleting some comments.
							Pos: conv.Args[0].End(),
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
					if !analyzerutil.FileUsesGoVersion(pass, astutil.EnclosingFile(curCall), versions.Go1_19) {
						continue
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

// mayFormatEmpty reports whether fmt.Sprintf might produce an empty string.
// It returns false in the following two cases:
//  1. formatStr contains non-operation characters.
//  2. formatStr contains formatting verbs besides s, v, x, X (verbs which may
//     produce empty results)
//
// In all other cases it returns true.
func mayFormatEmpty(formatStr string) bool {
	if formatStr == "" {
		return true
	}
	operations, err := fmtstr.Parse(formatStr, 0)
	if err != nil {
		// If formatStr is malformed, the printf analyzer will report a
		// diagnostic, so we can ignore this error.
		// Calling Parse on a string without % formatters also returns an error,
		// in which case we can safely return false.
		return false
	}
	totalOpsLen := 0
	for _, op := range operations {
		totalOpsLen += len(op.Text)
		if !strings.ContainsRune("svxX", rune(op.Verb.Verb)) && op.Prec.Fixed != 0 {
			// A non [s, v, x, X] formatter with non-zero precision cannot
			// produce an empty string.
			return false
		}
	}
	// If the format string contains non-operation characters, it cannot produce
	// the empty string.
	if totalOpsLen != len(formatStr) {
		return false
	}
	// If we get here, it means that all formatting verbs are %s, %v, %x, %X,
	// and there are no additional non-operation characters. We conservatively
	// report that this may format as an empty string, ignoring uses of
	// precision and the values of the formatter args.
	return true
}
