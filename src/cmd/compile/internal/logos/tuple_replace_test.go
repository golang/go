// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"strings"
	"testing"

	"cmd/compile/internal/syntax"
)

const replaceTupleSource = `package p

func f(xs []int) int {
	for _, x := range xs {
		if x > 1 {
			break
		}
		continue
	}
L:
	for {
		goto L
	}
	return 1 + 2
}
`

// TestReplaceTupleExprSkipsBranchTargets checks that the node swap finds a
// node that follows `break`, `continue`, and `goto` statements. With
// syntax.CheckBranches, each of them points at an enclosing statement, and
// following that pointer would walk the tree in a cycle.
func TestReplaceTupleExprSkipsBranchTargets(t *testing.T) {
	file, err := syntax.Parse(syntax.NewFileBase("p.go"), strings.NewReader(replaceTupleSource), nil, nil, syntax.CheckBranches)
	if err != nil {
		t.Fatal(err)
	}
	var old *syntax.Operation
	var ret *syntax.ReturnStmt
	targets := 0
	syntax.Inspect(file, func(n syntax.Node) bool {
		switch n := n.(type) {
		case *syntax.Operation:
			if n.Op == syntax.Add {
				old = n
			}
		case *syntax.ReturnStmt:
			ret = n
		case *syntax.BranchStmt:
			if n.Target != nil {
				targets++
			}
		}
		return true
	})
	if old == nil || ret == nil {
		t.Fatal("fixture lost its sum or its return")
	}
	if targets != 3 {
		t.Fatalf("break, continue, and goto carry %d targets, want 3", targets)
	}
	repl := nameOf(old.Pos(), "r")
	replaceTupleExpr(file, old, repl)
	if ret.Results != repl {
		t.Errorf("return holds %v, want the replacement", ret.Results)
	}
}
