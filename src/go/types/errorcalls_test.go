// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE ast.

package types_test

import (
	"go/ast"
	"go/token"
	"testing"
)

const errorfMinArgCount = 4

// TestErrorCalls makes sure that check.errorf calls have at least
// errorfMinArgCount arguments (otherwise we should use check.error).
func TestErrorCalls(t *testing.T) {
	fset := token.NewFileSet()
	files, err := pkgFiles(fset, ".")
	if err != nil {
		t.Fatal(err)
	}

	for _, file := range files {
		ast.Inspect(file, func(n ast.Node) bool {
			call, _ := n.(*ast.CallExpr)
			if call == nil {
				return true
			}
			selx, _ := call.Fun.(*ast.SelectorExpr)
			if selx == nil {
				return true
			}
			if !(isName(selx.X, "check") && isName(selx.Sel, "errorf")) {
				return true
			}
			// check.errorf calls should have at least errorfMinArgCount arguments:
			// position, code, format string, and arguments to format
			if n := len(call.Args); n < errorfMinArgCount {
				t.Errorf("%s: got %d arguments, want at least %d", fset.Position(call.Pos()), n, errorfMinArgCount)
				return false
			}
			return true
		})
	}
}

func isName(n ast.Node, name string) bool {
	if n, ok := n.(*ast.Ident); ok {
		return n.Name == name
	}
	return false
}
