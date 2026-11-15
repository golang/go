// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import "testing"

func TestLogosErrdeferParenFollowsStaysOrdinary(t *testing.T) {
	cases := []struct {
		name string
		src  string
	}{
		{"parenCallArg", "package p\nfunc f() int { return 1 }\nfunc g(errdefer func(int)) {\nerrdefer (f())\n}\n"},
		{"parenNameArg", "package p\nfunc g(errdefer func(int), x int) {\nerrdefer (x)\n}\n"},
		{"parenCallResult", "package p\nfunc g(errdefer func(int), x int) {\nerrdefer(x)\n}\n"},
		{"dotSelect", "package p\ntype T struct{}\nfunc (T) f() {}\nfunc g(errdefer T) {\nerrdefer.f()\n}\n"},
		{"define", "package p\nfunc g() {\nerrdefer := 1\n_ = errdefer\n}\n"},
	}
	for _, tc := range cases {
		f := parseLogos(t, tc.src)
		if got := countNodes[*ErrdeferStmt](f); got != 0 {
			t.Errorf("%s: found %d ErrdeferStmt nodes; identifier after errdefer must keep the ordinary meaning", tc.name, got)
		}
	}
	f := parseLogos(t, "package p\nfunc f() (int, error) { return 0, nil }\nfunc g() {\nerrdefer (f())\n}\n")
	stmts := funcBodyStmts(t, f, "g")
	if len(stmts) != 1 {
		t.Fatalf("parenCallArg: want 1 statement, found %d", len(stmts))
	}
	es, ok := stmts[0].(*ExprStmt)
	if !ok {
		t.Fatalf("parenCallArg: got %T, want *ExprStmt", stmts[0])
	}
	c, ok := es.X.(*CallExpr)
	if !ok {
		t.Fatalf("parenCallArg: got %T, want *CallExpr", es.X)
	}
	if n, ok := c.Fun.(*Name); !ok || n.Value != "errdefer" {
		t.Errorf("parenCallArg: call fun: got %v, want Name errdefer", c.Fun)
	}
}
