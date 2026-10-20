// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import (
	"bytes"
	"strings"
	"testing"
)

// Regression (SPEC Sec 5.2): a tuple literal needs a comma at its top
// level; a parenthesized expression without one must parse exactly as
// in Go. The defect: an element headed by a nested `(` ended at its
// own `)`, so `((a+b) + (c+d))` failed with "unexpected +, expected )".
func TestLogosParenExprVsTuple(t *testing.T) {
	f := parseLogos(t, "package p\n\nfunc f(a, b, c, d int) {\n\tx := ((a+b) + (c+d))\n\t_ = x\n}\n")
	stmts := funcBodyStmts(t, f, "f")
	if len(stmts) != 2 {
		t.Fatalf("got %d statements, want 2", len(stmts))
	}
	as, ok := stmts[0].(*AssignStmt)
	if !ok {
		t.Fatalf("first statement is %T, want *AssignStmt", stmts[0])
	}
	got := as.Rhs
	if _, isTuple := Unparen(got).(*TupleLit); isTuple {
		t.Fatalf("((a+b) + (c+d)) parsed as %T, want Operation (parenthesized expression)", got)
	}
	op, ok := Unparen(got).(*Operation)
	if !ok {
		t.Fatalf("((a+b) + (c+d)) parsed as %T, want *Operation", got)
	}
	if op.Op != Add {
		t.Fatalf("top-level op is %v, want Add", op.Op)
	}
	innerL, okL := Unparen(op.X).(*Operation)
	innerR, okR := Unparen(op.Y).(*Operation)
	if !okL || !okR || innerL.Op != Add || innerR.Op != Add {
		t.Fatalf("inner elements are %T/%T, want two parenthesized additions", op.X, op.Y)
	}
	// `(a)` and `((a))` stay ParenExpr around the name.
	for _, src := range []string{
		"package p\nfunc f(a int) { x := (a); _ = x }",
		"package p\nfunc f(a int) { x := ((a)); _ = x }",
	} {
		g := parseLogos(t, src)
		st := funcBodyStmts(t, g, "f")
		aa, ok := st[0].(*AssignStmt)
		if !ok {
			t.Fatalf("%s: first statement is %T, want *AssignStmt", src, st[0])
		}
		if _, isTuple := Unparen(aa.Rhs).(*TupleLit); isTuple {
			t.Fatalf("%s: parsed as *TupleLit, want ParenExpr", src)
		}
		if _, isParen := aa.Rhs.(*ParenExpr); !isParen {
			t.Fatalf("%s: rhs is %T, want *ParenExpr", src, aa.Rhs)
		}
	}
	// `(a, b)` is a tuple literal; `((a, b))` is a ParenExpr around it.
	g := parseLogos(t, "package p\nfunc f(a int, b string) { x := (a, b); y := ((a, b)); _, _ = x, y }")
	st := funcBodyStmts(t, g, "f")
	x, ok := st[0].(*AssignStmt)
	if !ok {
		t.Fatalf("first statement is %T, want *AssignStmt", st[0])
	}
	if _, isTuple := x.Rhs.(*TupleLit); !isTuple {
		t.Fatalf("(a, b) rhs is %T, want *TupleLit", x.Rhs)
	}
	y, ok := st[1].(*AssignStmt)
	if !ok {
		t.Fatalf("second statement is %T, want *AssignStmt", st[1])
	}
	py, isParen := y.Rhs.(*ParenExpr)
	if !isParen {
		t.Fatalf("((a, b)) rhs is %T, want *ParenExpr", y.Rhs)
	}
	if _, isTuple := py.X.(*TupleLit); !isTuple {
		t.Fatalf("((a, b)) inner is %T, want *TupleLit", py.X)
	}
}

// Neighbours of the regression: parenthesized operators, call
// arguments, both lambda shapes, and a tuple type in a var decl.
func TestLogosParenNeighbours(t *testing.T) {
	// `(a+b)*c` groups as an Operation, never a tuple.
	g := parseLogos(t, "package p\nfunc f(a, b, c int) { x := (a+b)*c; _ = x }")
	st := funcBodyStmts(t, g, "f")
	as, ok := st[0].(*AssignStmt)
	if !ok {
		t.Fatalf("(a+b)*c: first statement is %T, want *AssignStmt", st[0])
	}
	mul, ok := as.Rhs.(*Operation)
	if !ok || mul.Op != Mul {
		t.Fatalf("(a+b)*c rhs is %T, want *Operation Mul", as.Rhs)
	}
	if _, isTuple := Unparen(mul.X).(*TupleLit); isTuple {
		t.Fatalf("(a+b)*c left operand is a *TupleLit")
	}
	// `f((a+b), c)` parses as a two-argument call.
	g = parseLogos(t, "package p\nfunc f(a, b, c int) { g((a+b), c) }\nfunc g(x, y int) {}")
	st = funcBodyStmts(t, g, "f")
	cs, ok := st[0].(*ExprStmt)
	if !ok {
		t.Fatalf("f((a+b), c): first statement is %T, want *ExprStmt", st[0])
	}
	call, ok := cs.X.(*CallExpr)
	if !ok {
		t.Fatalf("f((a+b), c): statement expr is %T, want *CallExpr", cs.X)
	}
	if len(call.ArgList) != 2 {
		t.Fatalf("call has %d arguments, want 2", len(call.ArgList))
	}
	if _, isTuple := Unparen(call.ArgList[0]).(*TupleLit); isTuple {
		t.Fatalf("first call argument is a *TupleLit")
	}
	// `(a, b) => a+b` is a lambda; `(x) => x` is a one-parameter lambda.
	for _, src := range []string{
		"package p\nfunc f(a, b int) { g := (a, b) => a+b; _ = g }",
		"package p\nfunc f(a int) { g := (a) => a; _ = g }",
	} {
		h := parseLogos(t, src)
		st := funcBodyStmts(t, h, "f")
		aa, ok := st[0].(*AssignStmt)
		if !ok {
			t.Fatalf("%s: first statement is %T, want *AssignStmt", src, st[0])
		}
		if _, isLambda := aa.Rhs.(*LambdaExpr); !isLambda {
			t.Fatalf("%s: rhs is %T, want *LambdaExpr", src, aa.Rhs)
		}
	}
	// `(int, error)` in a var decl is a tuple type.
	h := parseLogos(t, "package p\nvar v (int, error)")
	var sawTuple bool
	Inspect(h, func(n Node) bool {
		if _, ok := n.(*TupleType); ok {
			sawTuple = true
			return false
		}
		return true
	})
	if !sawTuple {
		t.Fatalf("var v (int, error): no *TupleType in AST")
	}
}

// TestLogosParenMatchesGo is the differential attack on the shared `(`
// rule: every parenthesized expression below is valid Go without a
// top-level comma, so LogosMode must produce the identical tree (and
// positions) as plain mode, whatever token heads the element.
func TestLogosParenMatchesGo(t *testing.T) {
	exprs := []string{
		"(a)",
		"((a))",
		"(((a+b)))",
		"(a+b)*c",
		"((a+b) + (c+d))",
		"((a+b) * (c+d))",
		"((a+b) + c)",
		"(a + (b+c) + d)",
		"((a || b) && (c || d))",
		"(-a + b)",
		"(a.b + c.d)",
		"(f(x) + g(y))",
		"(a[i] + b[j])",
		"(x.(int) + 1)",
		"(*p)",
		"(*p + 1)",
		"(*f(x))",
		"(*p)(v)",
		"(*T)(v)",
		"(<-ch)",
		"(<-ch + 1)",
		"(<-chan int)(c)",
		"([]int{1}[0] + 1)",
		"(func() int { return 1 }() + 1)",
		"(map[string]int{\"a\": 1}[\"a\"] + 1)",
		"g((a+b), c)",
		"g(((a+b) + (c+d)), e)",
		"g(a, (*p + 1))",
	}
	for _, e := range exprs {
		src := "package p\nfunc f() { _ = " + e + " }\n"
		plain, perr := Parse(nil, strings.NewReader(src), nil, nil, CheckBranches)
		if perr != nil {
			t.Fatalf("%s: plain Go rejects the snippet: %v", e, perr)
		}
		var errs []string
		logos, _ := Parse(nil, strings.NewReader(src), func(err error) { errs = append(errs, err.Error()) }, nil, CheckBranches|LogosMode)
		if len(errs) != 0 {
			t.Errorf("%s: LogosMode errors %q, want none", e, errs)
			continue
		}
		var pd, ld bytes.Buffer
		if err := Fdump(&pd, plain); err != nil {
			t.Fatal(err)
		}
		if err := Fdump(&ld, logos); err != nil {
			t.Fatal(err)
		}
		if dumpAddr.ReplaceAllString(pd.String(), "0x") != dumpAddr.ReplaceAllString(ld.String(), "0x") {
			t.Errorf("%s: LogosMode tree differs from plain Go:\nplain:\n%s\nlogos:\n%s", e, pd.String(), ld.String())
		}
	}
}
