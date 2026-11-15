// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import "testing"

func countNodes[N Node](f *File) int {
	found := 0
	Inspect(f, func(n Node) bool {
		if _, ok := any(n).(N); ok {
			found++
		}
		return true
	})
	return found
}

func TestLogosKeywordPositionWins(t *testing.T) {
	tests := []struct {
		name  string
		src   string
		count func(*File) int
	}{
		{"errdefer", "package p\nfunc f(errdefer int) {\nerrdefer g()\n}\nfunc g() {}\n", func(f *File) int { return countNodes[*ErrdeferStmt](f) }},
		{"scope", "package p\nfunc f(scope int) {\nscope defer g()\n}\nfunc g() {}\n", func(f *File) int { return countNodes[*ScopeDeferStmt](f) }},
		{"group", "package p\nfunc f(group int, ctx int) {\ngo group (ctx) {\ngo g()\n}\n}\nfunc g() {}\n", func(f *File) int { return countNodes[*GroupStmt](f) }},
		{"yield", "package p\nfunc f(yield int) {\nyield yield\n}\n", func(f *File) int { return countNodes[*YieldStmt](f) }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := test.count(parseLogos(t, test.src)); got != 1 {
				t.Fatalf("%s: found %d keyword nodes, want 1", test.name, got)
			}
		})
	}
}

func TestLogosGapTokensStaySplit(t *testing.T) {
	f := parseLogos(t, "package p\nfunc f(o int) {\n_ = o? .Name\n}\n")
	sels := 0
	Inspect(f, func(n Node) bool {
		sel, ok := n.(*SelectorExpr)
		if !ok || sel.Sel.Value != "Name" {
			return true
		}
		sels++
		if _, ok := Unparen(sel.X).(*TryExpr); !ok {
			t.Errorf("o? .Name selector base is %T, want *TryExpr", sel.X)
		}
		return true
	})
	if sels != 1 {
		t.Errorf("o? .Name: found %d .Name selectors, want 1", sels)
	}
	if got := countNodes[*OptSelectExpr](f); got != 0 {
		t.Errorf("o? .Name: found %d *OptSelectExpr, want 0", got)
	}

	gmsgs := parseErrs(t, "package p\nfunc f(a, b int) {\n_ = a? ?b\n}\n", CheckBranches|LogosMode)
	if len(gmsgs) == 0 {
		t.Fatalf("a? ?b: parsed without error, want a syntax error (two postfix ? cannot merge into ??)")
	}
}

func TestLogosSpacedArrowRejected(t *testing.T) {
	msgs := parseErrs(t, "package p\nfunc f(a, b int) int {\ng := (a, b) = > a\nreturn g\n}\n", CheckBranches|LogosMode)
	if len(msgs) == 0 {
		t.Fatalf("(a, b) = > a: parsed without error, want a syntax error")
	}
}

func TestLogosQuestionNewlineEndsStatement(t *testing.T) {
	f := parseLogos(t, "package p\nfunc f() {\ng()?\nh()\n}\nfunc g() {}\nfunc h() {}\n")
	stmts := funcBodyStmts(t, f, "f")
	if len(stmts) != 2 {
		t.Fatalf("want 2 statements, found %d", len(stmts))
	}
	found := false
	Inspect(stmts[0], func(n Node) bool {
		if _, ok := n.(*TryExpr); ok {
			found = true
			return false
		}
		return true
	})
	if !found {
		t.Fatalf("first statement holds no *TryExpr")
	}
}

func TestLogosDotFloatKeepsSemicolon(t *testing.T) {
	f := parseLogos(t, "package p\nfunc f(x int) {\nx\n.5\n}\n")
	if got := len(funcBodyStmts(t, f, "f")); got != 2 {
		t.Fatalf("x newline .5: found %d statements, want 2", got)
	}
	msgs := parseErrs(t, "package p\nfunc f(x int) {\nx .5\n}\n", CheckBranches|LogosMode)
	if len(msgs) == 0 {
		t.Fatalf("x .5 on one line: parsed without error, want a syntax error")
	}
}

func TestLogosParenColonRejected(t *testing.T) {
	msgs := parseErrs(t, "package p\nfunc f(a int, b string) {\n_ = (a : int, b : string)\n}\n", CheckBranches|LogosMode)
	if len(msgs) == 0 {
		t.Fatalf("(a : int, b : string) in expression: parsed without error, want a syntax error")
	}
}
