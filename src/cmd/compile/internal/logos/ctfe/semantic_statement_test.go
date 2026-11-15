// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ctfe

import (
	"strings"
	"testing"
)

var (
	stmtIntT   = BasicType("int")
	stmtBoolT  = BasicType("bool")
	stmtSliceT = &Type{Kind: KSlice, Elem: stmtIntT}
)

type stmtFunc struct {
	src     string
	params  []*Type
	results []*Type
}

func stmtIntFn(src string, params ...*Type) stmtFunc {
	return stmtFunc{src: src, params: params, results: []*Type{stmtIntT}}
}

func stmtBoolFn(src string, params ...*Type) stmtFunc {
	return stmtFunc{src: src, params: params, results: []*Type{stmtBoolT}}
}

var stmtTruth = stmtIntFn("package p\nfunc Truth(b bool) int {\n\tif b {\n\t\treturn 1\n\t}\n\treturn 0\n}", stmtBoolT)

func stmtHost(t *testing.T, fns ...stmtFunc) *fakeHost {
	t.Helper()
	h := newFakeHost(t)
	for _, f := range fns {
		fn := parseFunc(t, f.src, true, f.params, f.results)
		h.funcs[fn.Name] = fn
	}
	return h
}

type stmtProbe struct {
	call string
	want int64
}

func stmtWantInt(t *testing.T, h *fakeHost, p stmtProbe) {
	t.Helper()
	v, _ := evalOne(t, h, Config{}, parseExpr(t, p.call))
	mustInt(t, v, p.want, p.call)
}

func TestSemanticStatementPredeclaredShadowing(t *testing.T) {
	for _, g := range []struct {
		name   string
		fn     stmtFunc
		probes []stmtProbe
	}{
		{"true parameter",
			stmtBoolFn("package p\nfunc T(true bool) bool { return true }", stmtBoolT),
			[]stmtProbe{{`Truth(T(false))`, 0}, {`Truth(T(true))`, 1}}},
		{"false parameter",
			stmtBoolFn("package p\nfunc F(false bool) bool { return false }", stmtBoolT),
			[]stmtProbe{{`Truth(F(true))`, 1}, {`Truth(F(false))`, 0}}},
		{"negated false parameter",
			stmtBoolFn("package p\nfunc N(false bool) bool { return !false }", stmtBoolT),
			[]stmtProbe{{`Truth(N(true))`, 0}, {`Truth(N(false))`, 1}}},
		{"true parameter as left operand of &&",
			stmtBoolFn("package p\nfunc A(true bool, x bool) bool { return true && x }", stmtBoolT, stmtBoolT),
			[]stmtProbe{{`Truth(A(false, true))`, 0}, {`Truth(A(true, true))`, 1}}},

		{"true parameter typed int",
			stmtIntFn("package p\nfunc P(true int) int { return true + 1 }", stmtIntT),
			[]stmtProbe{{`P(41)`, 42}}},
		{"nil parameter typed slice",
			stmtIntFn("package p\nfunc S(nil []int) int { return len(nil) }", stmtSliceT),
			[]stmtProbe{{`S([]int{1, 2, 3})`, 3}}},

		{"short declarations",
			stmtIntFn("package p\nfunc L(x int) int {\n\ttrue := x\n\tfalse := x + 1\n\tnil := x + 2\n\treturn true*100 + false*10 + nil\n}", stmtIntT),
			[]stmtProbe{{`L(1)`, 123}, {`L(4)`, 456}}},
		{"var declaration",
			stmtIntFn("package p\nfunc V() int {\n\tvar false = 7\n\treturn false + 1\n}"),
			[]stmtProbe{{`V()`, 8}}},
		{"const declaration",
			stmtIntFn("package p\nfunc C() int {\n\tconst nil = 40\n\treturn nil + 2\n}"),
			[]stmtProbe{{`C()`, 42}}},

		{"block shadow ends with its scope",
			stmtIntFn("package p\nfunc Scope(x bool) int {\n\tn := 0\n\tif x {\n\t\tfalse := 5\n\t\tn += false\n\t}\n\tif false {\n\t\tn += 100\n\t}\n\tif !false {\n\t\tn += 1000\n\t}\n\treturn n\n}", stmtBoolT),
			[]stmtProbe{{`Scope(true)`, 1005}, {`Scope(false)`, 1000}}},

		{"unshadowed true and false",
			stmtIntFn("package p\nfunc Unshadowed() int {\n\tn := 0\n\tif true {\n\t\tn += 1\n\t}\n\tif !false {\n\t\tn += 10\n\t}\n\treturn n\n}"),
			[]stmtProbe{{`Unshadowed()`, 11}}},
		{"unshadowed nil",
			stmtIntFn("package p\nfunc NilControl() int {\n\tvar a []int\n\tb := []int{1}\n\tn := 0\n\tif a == nil {\n\t\tn += 1\n\t}\n\tif b != nil {\n\t\tn += 20\n\t}\n\treturn n\n}"),
			[]stmtProbe{{`NilControl()`, 21}}},
	} {
		t.Run(g.name, func(t *testing.T) {
			h := stmtHost(t, g.fn, stmtTruth)
			for _, p := range g.probes {
				t.Run(p.call, func(t *testing.T) {
					stmtWantInt(t, h, p)
				})
			}
		})
	}
}

func TestSemanticStatementDiscardedCallResults(t *testing.T) {
	h := stmtHost(t,
		stmtFunc{
			src:    "package p\nfunc Set(s []int, i int, v int) { s[i] = v }",
			params: []*Type{stmtSliceT, stmtIntT, stmtIntT},
		},
		stmtFunc{
			src:     "package p\nfunc Tick(s []int) (int, bool) {\n\ts[0]++\n\treturn s[0], true\n}",
			params:  []*Type{stmtSliceT},
			results: []*Type{stmtIntT, stmtBoolT},
		},
		stmtFunc{
			src:     "package p\nfunc Boom(s []int) (int, int) { return s[5], 0 }",
			params:  []*Type{stmtSliceT},
			results: []*Type{stmtIntT, stmtIntT},
		},

		stmtIntFn("package p\nfunc ZeroResult() int {\n\ts := []int{0, 0, 0}\n\tSet(s, 0, 4)\n\tSet(s, 2, 5)\n\treturn s[0]*100 + s[1]*10 + s[2]\n}"),
		stmtIntFn("package p\nfunc MultiResult() int {\n\ts := []int{0}\n\tTick(s)\n\treturn s[0]\n}"),
		stmtIntFn("package p\nfunc Parenthesized() int {\n\ts := []int{0}\n\t(Tick(s))\n\t((Tick(s)))\n\treturn s[0]\n}"),
		stmtIntFn("package p\nfunc FirstResult() int {\n\ts := []int{0}\n\tx, _ := Tick(s)\n\treturn x\n}"),
		stmtIntFn("package p\nfunc SecondResult() int {\n\ts := []int{0}\n\t_, ok := Tick(s)\n\tif ok {\n\t\treturn s[0] + 10\n\t}\n\treturn -1\n}"),
		stmtIntFn("package p\nfunc Failing() int {\n\ts := []int{1}\n\tBoom(s)\n\treturn 1\n}"),
	)

	for _, p := range []stmtProbe{
		{`ZeroResult()`, 405},
		{`MultiResult()`, 1},
		{`Parenthesized()`, 2},
		{`FirstResult()`, 1},
		{`SecondResult()`, 11},
	} {
		t.Run(p.call, func(t *testing.T) {
			stmtWantInt(t, h, p)
		})
	}

	t.Run("failure inside a discarded call surfaces", func(t *testing.T) {
		cerr := evalErr(t, h, Config{}, parseExpr(t, `Failing()`))
		if cerr.Code != CodeEval || !strings.Contains(cerr.Detail, "index") {
			t.Fatalf("error = %v (code %v), want CodeEval index failure", cerr, cerr.Code)
		}
	})
}
