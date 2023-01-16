// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE ast.

package types2_test

import (
	"cmd/compile/internal/syntax"
	"testing"
)

const errorfMinArgCount = 4

// TestErrorCalls makes sure that check.errorf calls have at least
// errorfMinArgCount arguments (otherwise we should use check.error).
func TestErrorCalls(t *testing.T) {
	files, err := pkgFiles(".")
	if err != nil {
		t.Fatal(err)
	}

	for _, file := range files {
		syntax.Crawl(file, func(n syntax.Node) bool {
			call, _ := n.(*syntax.CallExpr)
			if call == nil {
				return false
			}
			selx, _ := call.Fun.(*syntax.SelectorExpr)
			if selx == nil {
				return false
			}
			if !(isName(selx.X, "check") && isName(selx.Sel, "errorf")) {
				return false
			}
			// check.errorf calls should have at least errorfMinArgCount arguments:
			// position, code, format string, and arguments to format
			if n := len(call.ArgList); n < errorfMinArgCount {
				t.Errorf("%s: got %d arguments, want at least %d", call.Pos(), n, errorfMinArgCount)
				return false
			}
			return false
		})
	}
}

func isName(n syntax.Node, name string) bool {
	if n, ok := n.(*syntax.Name); ok {
		return n.Value == name
	}
	return false
}
