// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"strings"
	"testing"

	"cmd/compile/internal/syntax"
)

// firstMapCall parses `var _ = expr` in LogosMode and returns the
// outermost call of a `.map` selector under it, with the file lowerer
// that holds the source.
func firstMapCall(t *testing.T, expr string) (*fileLower, *syntax.CallExpr) {
	t.Helper()
	data := []byte("package p\n\nvar _ = " + expr + "\n")
	file, err := syntax.Parse(syntax.NewFileBase("recv.lgo"), strings.NewReader(string(data)), nil, nil, syntax.LogosMode)
	if err != nil {
		t.Fatalf("parse %q: %v", expr, err)
	}
	var found *syntax.CallExpr
	syntax.Inspect(file, func(n syntax.Node) bool {
		call, ok := n.(*syntax.CallExpr)
		if !ok || found != nil {
			return found == nil
		}
		if sel, ok := call.Fun.(*syntax.SelectorExpr); ok && sel.Sel.Value == "map" {
			found = call
		}
		return found == nil
	})
	if found == nil {
		t.Fatalf("no .map call in %q", expr)
	}
	return &fileLower{src: data, start: lineStarts(data)}, found
}

// TestChainReceiverSpan pins the exact receiver and chain spans of the
// receiver shapes that syntax positions misplace: a slice expression
// sits at its `[`, a parenthesized receiver starts at its `(`, and a
// comment or a leading-dot line break can separate the receiver from
// the dot.
func TestChainReceiverSpan(t *testing.T) {
	cases := []struct{ name, expr, recv, whole string }{
		{"slice", "xs[1:].map(f)", "xs[1:]", "xs[1:].map(f)"},
		{"slice upto", "xs[:2].map(f)", "xs[:2]", "xs[:2].map(f)"},
		{"full slice", "xs[1:2:3].map(f)", "xs[1:2:3]", "xs[1:2:3].map(f)"},
		{"paren slice", "(xs[1:]).map(f)", "(xs[1:])", "(xs[1:]).map(f)"},
		{"paren name", "(xs).map(f)", "(xs)", "(xs).map(f)"},
		{"paren twice", "((xs)).map(f)", "((xs))", "((xs)).map(f)"},
		{"paren inside", "(xs).f[1:].map(f)", "(xs).f[1:]", "(xs).f[1:].map(f)"},
		{"paren comment", "( /* c */ xs[1:]).map(f)", "( /* c */ xs[1:])", "( /* c */ xs[1:]).map(f)"},
		{"paren own line", "(\n\txs[1:]).map(f)", "(\n\txs[1:])", "(\n\txs[1:]).map(f)"},
		{"index", `m["k"].map(f)`, `m["k"]`, `m["k"].map(f)`},
		{"index string", `m["a].b("].map(f)`, `m["a].b("]`, `m["a].b("].map(f)`},
		{"index slice", "g[1][2:].map(f)", "g[1][2:]", "g[1][2:].map(f)"},
		{"call slice", "f()[0:2].map(g)", "f()[0:2]", "f()[0:2].map(g)"},
		{"method slice", "x.m()[1:].map(f)", "x.m()[1:]", "x.m()[1:].map(f)"},
		{"assert slice", "a.([]int)[1:].map(f)", "a.([]int)[1:]", "a.([]int)[1:].map(f)"},
		{"literal slice", "[]int{1, 2}[1:].map(f)", "[]int{1, 2}[1:]", "[]int{1, 2}[1:].map(f)"},
		{"block comment", "xs[1:] /* c */ .map(f)", "xs[1:]", "xs[1:] /* c */ .map(f)"},
		{"comment with dot", "xs[1:] /* ].map( */ .map(f)", "xs[1:]", "xs[1:] /* ].map( */ .map(f)"},
		{"line comment dot", "xs[1:] // c\n\t.map(f)", "xs[1:]", "xs[1:] // c\n\t.map(f)"},
		{"leading dot", "xs[1:]\n\t.map(f)", "xs[1:]", "xs[1:]\n\t.map(f)"},
		{"multiline literal", "[]int{\n\t1,\n}[1:]\n\t.map(f)", "[]int{\n\t1,\n}[1:]", "[]int{\n\t1,\n}[1:]\n\t.map(f)"},
		{"nested receiver", "xs.map(g)[1:].map(f)", "xs.map(g)[1:]", "xs.map(g)[1:].map(f)"},
		{"nested leading dot", "xs[1:]\n\t.sortBy(g)[:2]\n\t.map(f)", "xs[1:]\n\t.sortBy(g)[:2]", "xs[1:]\n\t.sortBy(g)[:2]\n\t.map(f)"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			fl, call := firstMapCall(t, c.expr)
			recv := fl.receiverSpan(call)
			if got := string(fl.src[recv.start:recv.end]); got != c.recv {
				t.Errorf("receiver span %q, want %q", got, c.recv)
			}
			start, end := fl.chainSpan(call)
			if got := string(fl.src[start:end]); got != c.whole {
				t.Errorf("chain span %q, want %q", got, c.whole)
			}
		})
	}
}

// TestChainReceiverSpanInCall checks that a chain inside a call
// argument list takes only its own tokens: the call's opening
// parenthesis stays out, a parenthesized receiver keeps its pair, and
// a pair around the whole chain stays out.
func TestChainReceiverSpanInCall(t *testing.T) {
	cases := []struct{ name, expr, recv, whole string }{
		{"plain", "f(xs[1:].map(g))", "xs[1:]", "xs[1:].map(g)"},
		{"paren receiver", "f((xs[1:]).map(g))", "(xs[1:])", "(xs[1:]).map(g)"},
		{"paren receiver second", "f(a, (xs[1:]).map(g))", "(xs[1:])", "(xs[1:]).map(g)"},
		{"paren chain", "f((xs[1:].map(g)))", "xs[1:]", "xs[1:].map(g)"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			fl, call := firstMapCall(t, c.expr)
			recv := fl.receiverSpan(call)
			if got := string(fl.src[recv.start:recv.end]); got != c.recv {
				t.Errorf("receiver span %q, want %q", got, c.recv)
			}
			start, end := fl.chainSpan(call)
			if got := string(fl.src[start:end]); got != c.whole {
				t.Errorf("chain span %q, want %q", got, c.whole)
			}
		})
	}
}

// TestChainCommentsEjectOnce checks that the ejected text keeps a
// comment between the receiver and the dot, which the replaced chain
// would drop, and keeps a comment inside an argument once: the lambda
// body text already carries it.
func TestChainCommentsEjectOnce(t *testing.T) {
	src := []byte("package main\n\nfunc main() {\n\txs := []int{1, 2, 3}\n\t_ = xs[1:] /* gap */ .map(x => x /* body */ + 1)\n}\n")
	base := syntax.NewFileBase("comments.lgo")
	file := ejectParse(t, base, src, syntax.LogosMode|syntax.CheckBranches)
	pkg, info := ejectCheck(t, file)
	ejected, err := ApplyEdits(src, base, ejectLower(t, pkg, info, file, src))
	if err != nil {
		t.Fatalf("ApplyEdits: %v", err)
	}
	for _, c := range []string{"/* gap */", "/* body */"} {
		if n := strings.Count(string(ejected), c); n != 1 {
			t.Errorf("comment %s appears %d times in the ejected text, want 1:\n%s", c, n, ejected)
		}
	}
}

// TestFindCallParensNesting checks that the call's parentheses match
// across the semicolons of nested bodies and the brackets of its
// arguments: an explicit or automatic semicolon inside braces never ends
// the scan, and a string or comment holding a bracket or a semicolon
// stays out of the count.
func TestFindCallParensNesting(t *testing.T) {
	cases := []struct {
		name string
		src  string
		want int
	}{
		{"plain", "f(a, b)", 6},
		{"body semicolon", "f(func() int { y := 1; return y })", 33},
		{"for header", "f(func() { for i := 0; i < 1; i++ { g(i) } })", 44},
		{"index and slice", "f(xs[1:2], ys[0])", 16},
		{"literal braces", "f(T{1, 2}, \")\")", 14},
		{"string semicolon", "f(m[\"a;b\"])", 10},
		{"comment semicolon", "f(/* ; */ a)", 11},
		{"newline body", "f(func() {\n\ta()\n\tb()\n})", 22},
		{"first of two", "f(a).g(b)", 3},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			fl := &fileLower{src: []byte(c.src)}
			lparen, rparen := fl.findCallParens(1)
			if lparen != 1 || rparen != c.want {
				t.Errorf("findCallParens = (%d, %d), want (1, %d)", lparen, rparen, c.want)
			}
		})
	}
}

// TestChainCallEndParenFun checks that a call ends at the parenthesis
// that matches its own opening one, however its function operand is
// written: a parenthesized operand leaves a `)` before the argument
// list, and a function literal body in the arguments holds semicolons.
func TestChainCallEndParenFun(t *testing.T) {
	cases := []struct{ name, expr string }{
		{"paren name", "(f)(xs)"},
		{"paren selector", "(a.f)(xs, 1)"},
		{"paren call", "(g())(xs)"},
		{"body semicolon", "f(func() bool { y := 1; return y > 0 })"},
		{"paren literal", "(func(x int) int { y := x; return y })(1)"},
		{"plain", "f(xs)"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			data := []byte("package p\n\nvar _ = " + c.expr + "\n")
			file, err := syntax.Parse(syntax.NewFileBase("end.lgo"), strings.NewReader(string(data)), nil, nil, syntax.LogosMode)
			if err != nil {
				t.Fatalf("parse %q: %v", c.expr, err)
			}
			call, ok := file.DeclList[0].(*syntax.VarDecl).Values.(*syntax.CallExpr)
			if !ok {
				t.Fatalf("%q is not a call", c.expr)
			}
			fl := &fileLower{src: data, start: lineStarts(data)}
			want := strings.Index(string(data), c.expr) + len(c.expr)
			if got := fl.chainCallEnd(call); got != want {
				t.Errorf("chainCallEnd = %d, want %d (%q)", got, want, data[:got])
			}
		})
	}
}
