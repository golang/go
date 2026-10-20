// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import (
	"strings"
	"testing"
	"time"
)

// TestLogosParenNoBacktrack proves SPEC #38: `(` parses its element
// list once and then branches. A backtracking parser would re-parse
// every alternative at every level, so depth-500 nesting is infeasible
// for anything but a single linear pass; completing the parse is the
// proof. The 30s bound is orders of magnitude above the millisecond
// single-pass cost, so it only trips on algorithmic blowup.
func TestLogosParenNoBacktrack(t *testing.T) {
	const depth = 500
	var sb strings.Builder
	sb.WriteString("package p\nfunc f(x int) int {\n_ = ")
	for range depth {
		sb.WriteString("(")
	}
	sb.WriteString("x")
	for range depth {
		sb.WriteString(")")
	}
	sb.WriteString("\nreturn x\n}\n")
	start := time.Now()
	f := parseLogos(t, sb.String())
	if d := time.Since(start); d > 30*time.Second {
		t.Fatalf("depth-%d paren parse took %v, want well under 30s (single linear pass)", depth, d)
	}
	// Each level wraps exactly once around x: the single-element
	// branch ran at every level, never a tuple, never dropped.
	stmts := funcBodyStmts(t, f, "f")
	if len(stmts) != 2 {
		t.Fatalf("got %d statements, want 2 (_ = (...) and return)", len(stmts))
	}
	as, ok := stmts[0].(*AssignStmt)
	if !ok {
		t.Fatalf("first statement is %T, want *AssignStmt", stmts[0])
	}
	e := as.Rhs
	for range depth {
		px, ok := e.(*ParenExpr)
		if !ok {
			t.Fatalf("level unwrap got %T %v, want *ParenExpr", e, e)
		}
		e = px.X
	}
	name, ok := e.(*Name)
	if !ok || name.Value != "x" {
		t.Fatalf("innermost is %T %v, want Name x", e, e)
	}
}
