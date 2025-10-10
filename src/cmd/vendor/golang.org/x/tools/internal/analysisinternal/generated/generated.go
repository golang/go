// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package generated defines an analyzer whose result makes it
// convenient to skip diagnostics within generated files.
package generated

import (
	"go/ast"
	"go/token"
	"reflect"

	"golang.org/x/tools/go/analysis"
)

var Analyzer = &analysis.Analyzer{
	Name:       "generated",
	Doc:        "detect which Go files are generated",
	URL:        "https://pkg.go.dev/golang.org/x/tools/internal/analysisinternal/generated",
	ResultType: reflect.TypeFor[*Result](),
	Run: func(pass *analysis.Pass) (any, error) {
		set := make(map[*token.File]bool)
		for _, file := range pass.Files {
			if ast.IsGenerated(file) {
				set[pass.Fset.File(file.FileStart)] = true
			}
		}
		return &Result{fset: pass.Fset, generatedFiles: set}, nil
	},
}

type Result struct {
	fset           *token.FileSet
	generatedFiles map[*token.File]bool
}

// IsGenerated reports whether the position is within a generated file.
func (r *Result) IsGenerated(pos token.Pos) bool {
	return r.generatedFiles[r.fset.File(pos)]
}
