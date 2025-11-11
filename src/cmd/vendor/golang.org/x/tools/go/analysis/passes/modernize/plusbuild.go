// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modernize

import (
	"go/ast"
	"go/parser"
	"strings"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/internal/analysis/analyzerutil"
	"golang.org/x/tools/internal/goplsexport"
	"golang.org/x/tools/internal/versions"
)

var plusBuildAnalyzer = &analysis.Analyzer{
	Name: "plusbuild",
	Doc:  analyzerutil.MustExtractDoc(doc, "plusbuild"),
	URL:  "https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/modernize#plusbuild",
	Run:  plusbuild,
}

func init() {
	// Export to gopls until this is a published modernizer.
	goplsexport.PlusBuildModernizer = plusBuildAnalyzer
}

func plusbuild(pass *analysis.Pass) (any, error) {
	check := func(f *ast.File) {
		if !analyzerutil.FileUsesGoVersion(pass, f, versions.Go1_18) {
			return
		}

		// When gofmt sees a +build comment, it adds a
		// preceding equivalent //go:build directive, so in
		// formatted files we can assume that a +build line is
		// part of a comment group that starts with a
		// //go:build line and is followed by a blank line.
		//
		// While we cannot delete comments from an AST and
		// expect consistent output in general, this specific
		// case--deleting only some lines from a comment
		// block--does format correctly.
		for _, g := range f.Comments {
			sawGoBuild := false
			for _, c := range g.List {
				if sawGoBuild && strings.HasPrefix(c.Text, "// +build ") {
					pass.Report(analysis.Diagnostic{
						Pos:     c.Pos(),
						End:     c.End(),
						Message: "+build line is no longer needed",
						SuggestedFixes: []analysis.SuggestedFix{{
							Message: "Remove obsolete +build line",
							TextEdits: []analysis.TextEdit{{
								Pos: c.Pos(),
								End: c.End(),
							}},
						}},
					})
					break
				}
				if strings.HasPrefix(c.Text, "//go:build ") {
					sawGoBuild = true
				}
			}
		}
	}

	for _, f := range pass.Files {
		check(f)
	}
	for _, name := range pass.IgnoredFiles {
		if strings.HasSuffix(name, ".go") {
			f, err := parser.ParseFile(pass.Fset, name, nil, parser.ParseComments|parser.SkipObjectResolution)
			if err != nil {
				continue // parse error: ignore
			}
			check(f)
		}
	}
	return nil, nil
}
