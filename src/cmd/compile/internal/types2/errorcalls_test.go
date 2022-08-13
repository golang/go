// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE ast.

package types2_test

import (
	"cmd/compile/internal/syntax"
	"testing"
)

// TestErrorCalls makes sure that check.errorf calls have at
// least 3 arguments (otherwise we should be using check.error).
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
			// check.errorf calls should have more than 2 arguments:
			// position, format string, and arguments to format
			if n := len(call.ArgList); n <= 2 {
				t.Errorf("%s: got %d arguments, want > 2", call.Pos(), n)
				return true
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
