// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package typeindex defines an analyzer that provides a
// [golang.org/x/tools/internal/typesinternal/typeindex.Index].
//
// Like [golang.org/x/tools/go/analysis/passes/inspect], it is
// intended to be used as a helper by other analyzers; it reports no
// diagnostics of its own.
package typeindex

import (
	"reflect"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/internal/typesinternal/typeindex"
)

var Analyzer = &analysis.Analyzer{
	Name: "typeindex",
	Doc:  "indexes of type information for later passes",
	URL:  "https://pkg.go.dev/golang.org/x/tools/internal/analysis/typeindex",
	Run: func(pass *analysis.Pass) (any, error) {
		inspect := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)
		return typeindex.New(inspect, pass.Pkg, pass.TypesInfo), nil
	},
	RunDespiteErrors: true,
	Requires:         []*analysis.Analyzer{inspect.Analyzer},
	ResultType:       reflect.TypeFor[*typeindex.Index](),
}
