// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import (
	"bytes"
	"strings"
	"testing"
)

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

func TestLogosParenNeighbours(t *testing.T) {
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

func TestLogosParenColonMeansNamedType(t *testing.T) {
	f := parseLogos(t, "package p\nvar v (x: int, y: string)\n")
	decl, ok := f.DeclList[0].(*VarDecl)
	if !ok {
		t.Fatalf("decl 0 is %T; want *VarDecl", f.DeclList[0])
	}
	if decl.Values != nil {
		t.Fatalf("var decl has values %v; want none", decl.Values)
	}
	tt, ok := decl.Type.(*TupleType)
	if !ok {
		t.Fatalf("(x: int, y: string) is %T; want *TupleType", decl.Type)
	}
	if len(tt.Fields) != 2 || tt.Fields[0].Name == nil || tt.Fields[0].Name.Value != "x" || tt.Fields[1].Name == nil || tt.Fields[1].Name.Value != "y" {
		t.Fatalf("colon fields carry no names: %+v", tt.Fields)
	}
	src := "package p\nfunc f(a int) int {\nx := (a: 1)\nreturn x\n}\n"
	var msgs []string
	_, err := Parse(nil, strings.NewReader(src), func(err error) {
		msgs = append(msgs, err.Error())
	}, nil, CheckBranches|LogosMode)
	if err == nil && len(msgs) == 0 {
		t.Fatal("colon in value position parsed without error")
	}
	if len(msgs) == 0 || !strings.Contains(msgs[0], "expected )") {
		t.Fatalf("colon value first error is %q; want expected )", msgs)
	}
}

func TestLogosTupleNamedBracketElems(t *testing.T) {
	for _, tc := range []struct {
		src   string
		names []string
		types []string
	}{
		{"type A (id int, xs []int)", []string{"id", "xs"}, []string{"int", "[]int"}},
		{"type A (id int, p *int, a [2]int)", []string{"id", "p", "a"}, []string{"int", "*int", "[2]int"}},
		{"type A (xs []int?, n int)", []string{"xs", "n"}, []string{"[]int?", "int"}},
		{"type A (G[int], int)", []string{"", ""}, []string{"G[int]", "int"}},
		{"type A (G[int]?, x G[int])", []string{"", "x"}, []string{"G[int]?", "G[int]"}},
	} {
		f := parseLogos(t, "package p\n"+tc.src+"\n")
		decl, ok := f.DeclList[0].(*TypeDecl)
		if !ok {
			t.Fatalf("%s: decl 0 is %T; want *TypeDecl", tc.src, f.DeclList[0])
		}
		tt, ok := decl.Type.(*TupleType)
		if !ok || len(tt.Fields) != len(tc.names) {
			t.Fatalf("%s: type is %T; want *TupleType with %d fields", tc.src, decl.Type, len(tc.names))
		}
		for i, fld := range tt.Fields {
			name := ""
			if fld.Name != nil {
				name = fld.Name.Value
			}
			if got := String(fld.Type); name != tc.names[i] || got != tc.types[i] {
				t.Errorf("%s: field %d is %q %q; want %q %q", tc.src, i, name, got, tc.names[i], tc.types[i])
			}
		}
	}
}

func TestLogosParenOnceBranches(t *testing.T) {
	for _, tc := range []struct {
		shape  string
		src    string
		found  func(*File) bool
		errSub string
	}{
		{"paren", "package p\nfunc f(x int) { _ = (x) }\n", hasNode[*ParenExpr], ""},
		{"tuple", "package p\nfunc f(x int, y string) { _ = (x, y) }\n", hasNode[*TupleLit], ""},
		{"named", "package p\nvar v (x: int)\n", nil, "tuple type needs at least two"},
		{"namedpair", "package p\nvar v (x: int, y: string)\n", hasNode[*TupleType], ""},
		{"lambda", "package p\nfunc f(x int) { g := (x) => x; _ = g }\n", hasNode[*LambdaExpr], ""},
	} {
		t.Run(tc.shape, func(t *testing.T) {
			if tc.errSub != "" {
				var msgs []string
				_, _ = Parse(nil, strings.NewReader(tc.src), func(err error) {
					msgs = append(msgs, err.Error())
				}, nil, CheckBranches|LogosMode)
				for _, m := range msgs {
					if strings.Contains(m, tc.errSub) {
						return
					}
				}
				t.Errorf("%s: no error containing %q in %q", tc.shape, tc.errSub, msgs)
				return
			}
			if !tc.found(parseLogos(t, tc.src)) {
				t.Errorf("%s: no witness node in %q", tc.shape, tc.src)
			}
		})
	}
}

func TestLogosLambdaTypedParamNoPartialTree(t *testing.T) {
	for _, src := range []string{
		"package p\nfunc f() { g := (a int) => a + 1; _ = g }\n",
		"package p\nfunc f() { g := (a, b int) => a; _ = g }\n",
		"package p\nfunc f() { g := (a + b) => a; _ = g }\n",
	} {
		var msgs []string
		f, _ := Parse(nil, strings.NewReader(src), func(err error) {
			msgs = append(msgs, err.Error())
		}, nil, CheckBranches|LogosMode)
		found := false
		for _, m := range msgs {
			if strings.Contains(m, "lambda parameters must be names") {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("%q: no lambda parameter error in %q", src, msgs)
		}
		if hasNode[*LambdaExpr](f) {
			t.Errorf("%q: half-formed *LambdaExpr survives recovery", src)
		}
	}
}
