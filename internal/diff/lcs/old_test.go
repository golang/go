// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lcs

import (
	"math/rand"
	"testing"
)

func TestForwardOld(t *testing.T) {
	for _, tx := range Btests {
		lim := len(tx.a) + len(tx.b)
		left, right := []byte(tx.a), []byte(tx.b)
		g := newegraph(left, right, lim)
		lcs := g.forward()
		diffs := g.fromlcs(lcs)
		check(t, tx.a, lcs, tx.lcs)
		checkDiffs(t, tx.a, diffs, tx.b)

		g = newegraph(right, left, lim)
		lcs = g.forward()
		diffs = g.fromlcs(lcs)
		check(t, tx.b, lcs, tx.lcs)
		checkDiffs(t, tx.b, diffs, tx.a)
	}
}

func TestBackwardOld(t *testing.T) {
	for _, tx := range Btests {
		lim := len(tx.a) + len(tx.b)
		left, right := []byte(tx.a), []byte(tx.b)
		g := newegraph(left, right, lim)
		lcs := g.backward()
		check(t, tx.a, lcs, tx.lcs)
		diffs := g.fromlcs(lcs)
		checkDiffs(t, tx.a, diffs, tx.b)

		g = newegraph(right, left, lim)
		lcs = g.backward()
		diffs = g.fromlcs(lcs)
		check(t, tx.b, lcs, tx.lcs)
		checkDiffs(t, tx.b, diffs, tx.a)
	}
}

func TestTwosidedOld(t *testing.T) {
	// test both (a,b) and (b,a)
	for _, tx := range Btests {
		left, right := []byte(tx.a), []byte(tx.b)
		lim := len(tx.a) + len(tx.b)
		diffs, lcs := Compute(left, right, lim)
		check(t, tx.a, lcs, tx.lcs)
		checkDiffs(t, tx.a, diffs, tx.b)
		diffs, lcs = Compute(right, left, lim)
		check(t, tx.b, lcs, tx.lcs)
		checkDiffs(t, tx.b, diffs, tx.a)
	}
}

func TestIntOld(t *testing.T) {
	// need to avoid any characters in btests
	lfill, rfill := "AAAAAAAAAAAA", "BBBBBBBBBBBB"
	for _, tx := range Btests {
		if len(tx.a) < 2 || len(tx.b) < 2 {
			continue
		}
		left := []byte(tx.a + lfill)
		right := []byte(tx.b + rfill)
		lim := len(tx.a) + len(tx.b)
		diffs, lcs := Compute(left, right, lim)
		check(t, string(left), lcs, tx.lcs)
		checkDiffs(t, string(left), diffs, string(right))
		diffs, lcs = Compute(right, left, lim)
		check(t, string(right), lcs, tx.lcs)
		checkDiffs(t, string(right), diffs, string(left))

		left = []byte(lfill + tx.a)
		right = []byte(rfill + tx.b)
		diffs, lcs = Compute(left, right, lim)
		check(t, string(left), lcs, tx.lcs)
		checkDiffs(t, string(left), diffs, string(right))
		diffs, lcs = Compute(right, left, lim)
		check(t, string(right), lcs, tx.lcs)
		checkDiffs(t, string(right), diffs, string(left))
	}
}

func TestSpecialOld(t *testing.T) { // needs lcs.fix
	a := []byte("golang.org/x/tools/intern")
	b := []byte("github.com/google/safehtml/template\"\n\t\"golang.org/x/tools/intern")
	diffs, lcs := Compute(a, b, 4)
	if !lcs.valid() {
		t.Errorf("%d,%v", len(diffs), lcs)
	}
}

func TestRegressionOld001(t *testing.T) {
	a := "// Copyright 2019 The Go Authors. All rights reserved.\n// Use of this source code is governed by a BSD-style\n// license that can be found in the LICENSE file.\n\npackage diff_test\n\nimport (\n\t\"fmt\"\n\t\"math/rand\"\n\t\"strings\"\n\t\"testing\"\n\n\t\"golang.org/x/tools/gopls/internal/lsp/diff\"\n\t\"golang.org/x/tools/internal/diff/difftest\"\n\t\"golang.org/x/tools/gopls/internal/span\"\n)\n"

	b := "// Copyright 2019 The Go Authors. All rights reserved.\n// Use of this source code is governed by a BSD-style\n// license that can be found in the LICENSE file.\n\npackage diff_test\n\nimport (\n\t\"fmt\"\n\t\"math/rand\"\n\t\"strings\"\n\t\"testing\"\n\n\t\"github.com/google/safehtml/template\"\n\t\"golang.org/x/tools/gopls/internal/lsp/diff\"\n\t\"golang.org/x/tools/internal/diff/difftest\"\n\t\"golang.org/x/tools/gopls/internal/span\"\n)\n"
	for i := 1; i < len(b); i++ {
		diffs, lcs := Compute([]byte(a), []byte(b), int(i)) // 14 from gopls
		if !lcs.valid() {
			t.Errorf("%d,%v", len(diffs), lcs)
		}
		checkDiffs(t, a, diffs, b)
	}
}

func TestRegressionOld002(t *testing.T) {
	a := "n\"\n)\n"
	b := "n\"\n\t\"golang.org/x//nnal/stack\"\n)\n"
	for i := 1; i <= len(b); i++ {
		diffs, lcs := Compute([]byte(a), []byte(b), int(i))
		if !lcs.valid() {
			t.Errorf("%d,%v", len(diffs), lcs)
		}
		checkDiffs(t, a, diffs, b)
	}
}

func TestRegressionOld003(t *testing.T) {
	a := "golang.org/x/hello v1.0.0\nrequire golang.org/x/unused v1"
	b := "golang.org/x/hello v1"
	for i := 1; i <= len(a); i++ {
		diffs, lcs := Compute([]byte(a), []byte(b), int(i))
		if !lcs.valid() {
			t.Errorf("%d,%v", len(diffs), lcs)
		}
		checkDiffs(t, a, diffs, b)
	}
}

func TestRandOld(t *testing.T) {
	rand.Seed(1)
	for i := 0; i < 1000; i++ {
		a := []rune(randstr("abω", 16))
		b := []rune(randstr("abωc", 16))
		g := newegraph(a, b, 24) // large enough to get true lcs
		two := g.twosided()
		forw := g.forward()
		back := g.backward()
		if lcslen(two) != lcslen(forw) || lcslen(forw) != lcslen(back) {
			t.Logf("\n%v\n%v\n%v", forw, back, two)
			t.Fatalf("%d forw:%d back:%d two:%d", i, lcslen(forw), lcslen(back), lcslen(two))
		}
		if !two.valid() || !forw.valid() || !back.valid() {
			t.Errorf("check failure")
		}
	}
}

func BenchmarkTwoOld(b *testing.B) {
	tests := genBench("abc", 96)
	for i := 0; i < b.N; i++ {
		for _, tt := range tests {
			_, two := Compute([]byte(tt.before), []byte(tt.after), 100)
			if !two.valid() {
				b.Error("check failed")
			}
		}
	}
}

func BenchmarkForwOld(b *testing.B) {
	tests := genBench("abc", 96)
	for i := 0; i < b.N; i++ {
		for _, tt := range tests {
			_, two := Compute([]byte(tt.before), []byte(tt.after), 100)
			if !two.valid() {
				b.Error("check failed")
			}
		}
	}
}

func genBench(set string, n int) []struct{ before, after string } {
	// before and after for benchmarks. 24 strings of length n with
	// before and after differing at least once, and about 5%
	rand.Seed(3)
	var ans []struct{ before, after string }
	for i := 0; i < 24; i++ {
		// maybe b should have an approximately known number of diffs
		a := randstr(set, n)
		cnt := 0
		bb := make([]rune, 0, n)
		for _, r := range a {
			if rand.Float64() < .05 {
				cnt++
				r = 'N'
			}
			bb = append(bb, r)
		}
		if cnt == 0 {
			// avoid == shortcut
			bb[n/2] = 'N'
		}
		ans = append(ans, struct{ before, after string }{a, string(bb)})
	}
	return ans
}
