// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import (
	"strings"
	"testing"
)

// M1-S2 parser attacks for TryExpr{X Expr} and ErrdeferStmt{Call Expr}
// (logos-docs/IMPLEMENTATION.md).
//
// RED: neither node exists yet, so this file does not compile
// (undefined: TryExpr, undefined: ErrdeferStmt) and every `?`/errdefer
// form below fails to parse.

func parseLogos(t *testing.T, src string) *File {
	t.Helper()
	var msgs []string
	f, err := Parse(nil, strings.NewReader(src), func(err error) {
		msgs = append(msgs, err.Error())
	}, nil, CheckBranches|LogosMode)
	if err != nil {
		t.Fatalf("parse %q: %v", src, err)
	}
	if len(msgs) != 0 {
		t.Fatalf("parse %q: unexpected errors: %q", src, msgs)
	}
	return f
}

func printFull(t *testing.T, n Node) string {
	t.Helper()
	var buf strings.Builder
	// Form 0 is the default full form: unlike String (ShortForm), it
	// keeps function bodies instead of printing them as `{…}`.
	if _, err := Fprint(&buf, n, 0); err != nil {
		t.Fatalf("print: %v", err)
	}
	return buf.String()
}

func checkPos(t *testing.T, what string, p Pos, line, col uint) {
	t.Helper()
	if p.Line() != line || p.Col() != col {
		t.Errorf("%s: got %d:%d, want %d:%d", what, p.Line(), p.Col(), line, col)
	}
}

func funcBodyStmts(t *testing.T, f *File, name string) []Stmt {
	t.Helper()
	for _, d := range f.DeclList {
		if fd, ok := d.(*FuncDecl); ok && fd.Name.Value == name {
			if fd.Body == nil {
				t.Fatalf("func %s has no body", name)
			}
			return fd.Body.List
		}
	}
	t.Fatalf("func %s not found", name)
	return nil
}

func TestLogosTryExprPositions(t *testing.T) {
	// Columns of "func f() { g()? }": g at 2:12, ? at 2:15.
	f := parseLogos(t, "package p\nfunc f() { g()? }\n")
	var tries []*TryExpr
	Inspect(f, func(n Node) bool {
		if te, ok := n.(*TryExpr); ok {
			tries = append(tries, te)
		}
		return true
	})
	if len(tries) != 1 {
		t.Fatalf("want 1 *TryExpr, found %d", len(tries))
	}
	try := tries[0]
	// Postfix house style (SelectorExpr `a@.b`, CallExpr `f@()` in
	// nodes_test.go): each node takes its operator position, so the
	// TryExpr sits at `?` (2:15) and its CallExpr operand at `(` (2:13).
	checkPos(t, "TryExpr.Pos", try.Pos(), 2, 15)
	x, ok := try.X.(*CallExpr)
	if !ok {
		t.Fatalf("TryExpr.X: got %T, want *CallExpr", try.X)
	}
	checkPos(t, "TryExpr.X.Pos", x.Pos(), 2, 13)
	checkPos(t, "StartPos(TryExpr)", StartPos(try), 2, 12)
	checkPos(t, "EndPos(TryExpr.X)", EndPos(try.X), 2, 13)
	// One past the last token: `?` is at 2:15, mirroring the Name
	// `p.Col()+uint(len(n.Value))` precedent in EndPos.
	checkPos(t, "EndPos(TryExpr)", EndPos(try), 2, 16)
}

func TestLogosErrdeferStmtPositions(t *testing.T) {
	// Columns of "func f() { errdefer g() }": errdefer at 2:12, g at 2:21.
	f := parseLogos(t, "package p\nfunc f() { errdefer g() }\n")
	stmts := funcBodyStmts(t, f, "f")
	if len(stmts) != 1 {
		t.Fatalf("want 1 statement, found %d", len(stmts))
	}
	es, ok := stmts[0].(*ErrdeferStmt)
	if !ok {
		t.Fatalf("got %T, want *ErrdeferStmt", stmts[0])
	}
	checkPos(t, "ErrdeferStmt.Pos", es.Pos(), 2, 12)
	call, ok := es.Call.(*CallExpr)
	if !ok {
		t.Fatalf("ErrdeferStmt.Call: got %T, want *CallExpr", es.Call)
	}
	checkPos(t, "ErrdeferStmt.Call.Pos", call.Pos(), 2, 22)
	checkPos(t, "StartPos(ErrdeferStmt)", StartPos(es), 2, 12)
	checkPos(t, "EndPos(ErrdeferStmt)", EndPos(es), 2, 22)

	// Columns of "func f() { errdefer g()? }": ? at 2:24.
	f2 := parseLogos(t, "package p\nfunc f() { errdefer g()? }\n")
	stmts2 := funcBodyStmts(t, f2, "f")
	if len(stmts2) != 1 {
		t.Fatalf("want 1 statement, found %d", len(stmts2))
	}
	es2, ok := stmts2[0].(*ErrdeferStmt)
	if !ok {
		t.Fatalf("got %T, want *ErrdeferStmt", stmts2[0])
	}
	tc, ok := es2.Call.(*TryExpr)
	if !ok {
		t.Fatalf("ErrdeferStmt.Call: got %T, want *TryExpr", es2.Call)
	}
	checkPos(t, "ErrdeferStmt(?).Call.Pos", tc.Pos(), 2, 24)
	if _, ok := tc.X.(*CallExpr); !ok {
		t.Fatalf("TryExpr.X: got %T, want *CallExpr", tc.X)
	}
}

func TestLogosPrinterRoundTrip(t *testing.T) {
	const src = "package p\nfunc f() { g()? }\nfunc h() { errdefer g() }\n"
	f := parseLogos(t, src)
	out := printFull(t, f)
	if !strings.Contains(out, "g()?") {
		t.Errorf("printed output missing X?: %q", out)
	}
	if !strings.Contains(out, "errdefer g()") {
		t.Errorf("printed output missing errdefer call: %q", out)
	}
	var msgs []string
	again, err := Parse(nil, strings.NewReader(out), func(err error) {
		msgs = append(msgs, err.Error())
	}, nil, CheckBranches|LogosMode)
	if err != nil || len(msgs) != 0 {
		t.Fatalf("reparse printed output %q: err=%v msgs=%q", out, err, msgs)
	}
	if out2 := printFull(t, again); out2 != out {
		t.Errorf("printer not idempotent:\nfirst:  %q\nsecond: %q", out, out2)
	}
}

func TestLogosWalkVisitsLogosNodes(t *testing.T) {
	f := parseLogos(t, "package p\nfunc f() { g()?; errdefer h() }\n")
	var tries []*TryExpr
	var edefs []*ErrdeferStmt
	var calls []*CallExpr
	var order []string
	Inspect(f, func(n Node) bool {
		switch n := n.(type) {
		case *TryExpr:
			tries = append(tries, n)
			order = append(order, "try")
		case *ErrdeferStmt:
			edefs = append(edefs, n)
			order = append(order, "errdefer")
		case *CallExpr:
			calls = append(calls, n)
			order = append(order, "call")
		}
		return true
	})
	if len(tries) != 1 || len(edefs) != 1 || len(calls) != 2 {
		t.Fatalf("want 1 try, 1 errdefer, 2 calls; got %d, %d, %d (%q)",
			len(tries), len(edefs), len(calls), order)
	}
	want := []string{"try", "call", "errdefer", "call"}
	if strings.Join(order, ",") != strings.Join(want, ",") {
		t.Errorf("walk order: got %q, want %q", order, want)
	}
	if tries[0].X != calls[0] {
		t.Errorf("walk did not visit TryExpr.X")
	}
	if edefs[0].Call != calls[1] {
		t.Errorf("walk did not visit ErrdeferStmt.Call")
	}
}

func TestLogosErrdeferKeepsGoMeaning(t *testing.T) {
	tests := []struct {
		name  string
		src   string
		check func(t *testing.T, f *File)
	}{
		{
			name: "define",
			src:  "package p\nfunc f() {\nerrdefer := 1\n_ = errdefer\n}\n",
			check: func(t *testing.T, f *File) {
				stmts := funcBodyStmts(t, f, "f")
				if len(stmts) != 2 {
					t.Fatalf("want 2 statements, found %d", len(stmts))
				}
				if _, ok := stmts[0].(*AssignStmt); !ok {
					t.Errorf("errdefer := 1: got %T, want *AssignStmt", stmts[0])
				}
			},
		},
		{
			name: "assign",
			src:  "package p\nfunc f() {\nerrdefer := 1\nerrdefer = 2\n_ = errdefer\n}\n",
			check: func(t *testing.T, f *File) {
				stmts := funcBodyStmts(t, f, "f")
				if len(stmts) != 3 {
					t.Fatalf("want 3 statements, found %d", len(stmts))
				}
				for i, s := range stmts {
					if _, ok := s.(*AssignStmt); !ok {
						t.Errorf("stmt %d: got %T, want *AssignStmt", i, s)
					}
				}
			},
		},
		{
			name: "call",
			src:  "package p\nfunc f(errdefer func(int), x int) {\nerrdefer(x)\n}\n",
			check: func(t *testing.T, f *File) {
				stmts := funcBodyStmts(t, f, "f")
				if len(stmts) != 1 {
					t.Fatalf("want 1 statement, found %d", len(stmts))
				}
				es, ok := stmts[0].(*ExprStmt)
				if !ok {
					t.Fatalf("got %T, want *ExprStmt", stmts[0])
				}
				c, ok := es.X.(*CallExpr)
				if !ok {
					t.Fatalf("got %T, want *CallExpr", es.X)
				}
				if n, ok := c.Fun.(*Name); !ok || n.Value != "errdefer" {
					t.Errorf("call fun: got %v, want Name errdefer", c.Fun)
				}
			},
		},
		{
			name: "select",
			src:  "package p\ntype T struct{}\nfunc (T) f() {}\nfunc g(errdefer T) {\nerrdefer.f()\n}\n",
			check: func(t *testing.T, f *File) {
				stmts := funcBodyStmts(t, f, "g")
				if len(stmts) != 1 {
					t.Fatalf("want 1 statement, found %d", len(stmts))
				}
				es, ok := stmts[0].(*ExprStmt)
				if !ok {
					t.Fatalf("got %T, want *ExprStmt", stmts[0])
				}
				c, ok := es.X.(*CallExpr)
				if !ok {
					t.Fatalf("got %T, want *CallExpr", es.X)
				}
				sel, ok := c.Fun.(*SelectorExpr)
				if !ok || sel.Sel.Value != "f" {
					t.Errorf("call fun: got %v, want selector .f", c.Fun)
				}
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			f := parseLogos(t, test.src)
			var nErrdefer int
			Inspect(f, func(n Node) bool {
				if _, ok := n.(*ErrdeferStmt); ok {
					nErrdefer++
				}
				return true
			})
			if nErrdefer != 0 {
				t.Errorf("want no *ErrdeferStmt, found %d", nErrdefer)
			}
			test.check(t, f)
		})
	}
}
