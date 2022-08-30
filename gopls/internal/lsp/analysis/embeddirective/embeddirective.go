// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package embeddirective defines an Analyzer that validates import for //go:embed directive.
package embeddirective

import (
	"go/ast"
	"strings"

	"golang.org/x/tools/go/analysis"
)

const Doc = `check for //go:embed directive import

This analyzer checks that the embed package is imported when source code contains //go:embed comment directives.
The embed package must be imported for //go:embed directives to function.import _ "embed".`

var Analyzer = &analysis.Analyzer{
	Name:             "embed",
	Doc:              Doc,
	Requires:         []*analysis.Analyzer{},
	Run:              run,
	RunDespiteErrors: true,
}

func run(pass *analysis.Pass) (interface{}, error) {
	for _, f := range pass.Files {
		com := hasEmbedDirectiveComment(f)
		if com != nil {
			assertEmbedImport(pass, com, f)
		}
	}
	return nil, nil
}

// Check if the comment contains //go:embed directive.
func hasEmbedDirectiveComment(f *ast.File) *ast.Comment {
	for _, cg := range f.Comments {
		for _, c := range cg.List {
			if strings.HasPrefix(c.Text, "//go:embed ") {
				return c
			}
		}
	}
	return nil
}

// Verifies that "embed" import exists for //go:embed directive.
func assertEmbedImport(pass *analysis.Pass, com *ast.Comment, f *ast.File) {
	for _, imp := range f.Imports {
		if "\"embed\"" == imp.Path.Value {
			return
		}
	}
	pass.Report(analysis.Diagnostic{Pos: com.Pos(), End: com.Pos() + 10, Message: "The \"embed\" package must be imported when using go:embed directives."})
}
