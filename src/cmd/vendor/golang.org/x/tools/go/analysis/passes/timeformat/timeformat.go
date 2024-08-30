// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package timeformat defines an Analyzer that checks for the use
// of time.Format or time.Parse calls with a bad format.
package timeformat

import (
	_ "embed"
	"go/ast"
	"go/constant"
	"go/token"
	"go/types"
	"strings"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/analysis/passes/internal/analysisutil"
	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/go/types/typeutil"
)

const badFormat = "2006-02-01"
const goodFormat = "2006-01-02"

//go:embed doc.go
var doc string

var Analyzer = &analysis.Analyzer{
	Name:     "timeformat",
	Doc:      analysisutil.MustExtractDoc(doc, "timeformat"),
	URL:      "https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/timeformat",
	Requires: []*analysis.Analyzer{inspect.Analyzer},
	Run:      run,
}

func run(pass *analysis.Pass) (interface{}, error) {
	// Note: (time.Time).Format is a method and can be a typeutil.Callee
	// without directly importing "time". So we cannot just skip this package
	// when !analysisutil.Imports(pass.Pkg, "time").
	// TODO(taking): Consider using a prepass to collect typeutil.Callees.

	inspect := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)

	nodeFilter := []ast.Node{
		(*ast.CallExpr)(nil),
	}
	inspect.Preorder(nodeFilter, func(n ast.Node) {
		call := n.(*ast.CallExpr)
		fn, ok := typeutil.Callee(pass.TypesInfo, call).(*types.Func)
		if !ok {
			return
		}
		if !isTimeDotFormat(fn) && !isTimeDotParse(fn) {
			return
		}
		if len(call.Args) > 0 {
			arg := call.Args[0]
			badAt := badFormatAt(pass.TypesInfo, arg)

			if badAt > -1 {
				// Check if it's a literal string, otherwise we can't suggest a fix.
				if _, ok := arg.(*ast.BasicLit); ok {
					pos := int(arg.Pos()) + badAt + 1 // +1 to skip the " or `
					end := pos + len(badFormat)

					pass.Report(analysis.Diagnostic{
						Pos:     token.Pos(pos),
						End:     token.Pos(end),
						Message: badFormat + " should be " + goodFormat,
						SuggestedFixes: []analysis.SuggestedFix{{
							Message: "Replace " + badFormat + " with " + goodFormat,
							TextEdits: []analysis.TextEdit{{
								Pos:     token.Pos(pos),
								End:     token.Pos(end),
								NewText: []byte(goodFormat),
							}},
						}},
					})
				} else {
					pass.Reportf(arg.Pos(), badFormat+" should be "+goodFormat)
				}
			}
		}
	})
	return nil, nil
}

func isTimeDotFormat(f *types.Func) bool {
	if f.Name() != "Format" || f.Pkg() == nil || f.Pkg().Path() != "time" {
		return false
	}
	// Verify that the receiver is time.Time.
	recv := f.Type().(*types.Signature).Recv()
	return recv != nil && analysisutil.IsNamedType(recv.Type(), "time", "Time")
}

func isTimeDotParse(f *types.Func) bool {
	return analysisutil.IsFunctionNamed(f, "time", "Parse")
}

// badFormatAt return the start of a bad format in e or -1 if no bad format is found.
func badFormatAt(info *types.Info, e ast.Expr) int {
	tv, ok := info.Types[e]
	if !ok { // no type info, assume good
		return -1
	}

	t, ok := tv.Type.(*types.Basic) // sic, no unalias
	if !ok || t.Info()&types.IsString == 0 {
		return -1
	}

	if tv.Value == nil {
		return -1
	}

	return strings.Index(constant.StringVal(tv.Value), badFormat)
}
