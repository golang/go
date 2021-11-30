// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package infertypeargs defines an analyzer that checks for explicit function
// arguments that could be inferred.
package infertypeargs

import (
	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
)

const Doc = `check for unnecessary type arguments in call expressions

Explicit type arguments may be omitted from call expressions if they can be
inferred from function arguments, or from other type arguments:

	func f[T any](T) {}
	
	func _() {
		f[string]("foo") // string could be inferred
	}
`

var Analyzer = &analysis.Analyzer{
	Name:     "infertypeargs",
	Doc:      Doc,
	Requires: []*analysis.Analyzer{inspect.Analyzer},
	Run:      run,
}
