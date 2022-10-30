// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lcs

import (
	"fmt"
	"io/ioutil"
	"log"
	"math/rand"
	"strings"
	"testing"
)

func TestAlgosOld(t *testing.T) {
	for i, algo := range []func(*editGraph) lcs{forward, backward, twosided} {
		t.Run(strings.Fields("forward backward twosided")[i], func(t *testing.T) {
			for _, tx := range Btests {
				lim := len(tx.a) + len(tx.b)

				diffs, lcs := compute(stringSeqs{tx.a, tx.b}, algo, lim)
				check(t, tx.a, lcs, tx.lcs)
				checkDiffs(t, tx.a, diffs, tx.b)

				diffs, lcs = compute(stringSeqs{tx.b, tx.a}, algo, lim)
				check(t, tx.b, lcs, tx.lcs)
				checkDiffs(t, tx.b, diffs, tx.a)
			}
		})
	}
}

func TestIntOld(t *testing.T) {
	// need to avoid any characters in btests
	lfill, rfill := "AAAAAAAAAAAA", "BBBBBBBBBBBB"
	for _, tx := range Btests {
		if len(tx.a) < 2 || len(tx.b) < 2 {
			continue
		}
		left := tx.a + lfill
		right := tx.b + rfill
		lim := len(tx.a) + len(tx.b)
		diffs, lcs := compute(stringSeqs{left, right}, twosided, lim)
		check(t, left, lcs, tx.lcs)
		checkDiffs(t, left, diffs, right)
		diffs, lcs = compute(stringSeqs{right, left}, twosided, lim)
		check(t, right, lcs, tx.lcs)
		checkDiffs(t, right, diffs, left)

		left = lfill + tx.a
		right = rfill + tx.b
		diffs, lcs = compute(stringSeqs{left, right}, twosided, lim)
		check(t, left, lcs, tx.lcs)
		checkDiffs(t, left, diffs, right)
		diffs, lcs = compute(stringSeqs{right, left}, twosided, lim)
		check(t, right, lcs, tx.lcs)
		checkDiffs(t, right, diffs, left)
	}
}

func TestSpecialOld(t *testing.T) { // exercises lcs.fix
	a := "golang.org/x/tools/intern"
	b := "github.com/google/safehtml/template\"\n\t\"golang.org/x/tools/intern"
	diffs, lcs := compute(stringSeqs{a, b}, twosided, 4)
	if !lcs.valid() {
		t.Errorf("%d,%v", len(diffs), lcs)
	}
}

func TestRegressionOld001(t *testing.T) {
	a := "// Copyright 2019 The Go Authors. All rights reserved.\n// Use of this source code is governed by a BSD-style\n// license that can be found in the LICENSE file.\n\npackage diff_test\n\nimport (\n\t\"fmt\"\n\t\"math/rand\"\n\t\"strings\"\n\t\"testing\"\n\n\t\"golang.org/x/tools/gopls/internal/lsp/diff\"\n\t\"golang.org/x/tools/internal/diff/difftest\"\n\t\"golang.org/x/tools/gopls/internal/span\"\n)\n"

	b := "// Copyright 2019 The Go Authors. All rights reserved.\n// Use of this source code is governed by a BSD-style\n// license that can be found in the LICENSE file.\n\npackage diff_test\n\nimport (\n\t\"fmt\"\n\t\"math/rand\"\n\t\"strings\"\n\t\"testing\"\n\n\t\"github.com/google/safehtml/template\"\n\t\"golang.org/x/tools/gopls/internal/lsp/diff\"\n\t\"golang.org/x/tools/internal/diff/difftest\"\n\t\"golang.org/x/tools/gopls/internal/span\"\n)\n"
	for i := 1; i < len(b); i++ {
		diffs, lcs := compute(stringSeqs{a, b}, twosided, i) // 14 from gopls
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
		diffs, lcs := compute(stringSeqs{a, b}, twosided, i)
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
		diffs, lcs := compute(stringSeqs{a, b}, twosided, i)
		if !lcs.valid() {
			t.Errorf("%d,%v", len(diffs), lcs)
		}
		checkDiffs(t, a, diffs, b)
	}
}

func TestRandOld(t *testing.T) {
	rand.Seed(1)
	for i := 0; i < 1000; i++ {
		// TODO(adonovan): use ASCII and bytesSeqs here? The use of
		// non-ASCII isn't relevant to the property exercised by the test.
		a := []rune(randstr("abω", 16))
		b := []rune(randstr("abωc", 16))
		seq := runesSeqs{a, b}

		const lim = 24 // large enough to get true lcs
		_, forw := compute(seq, forward, lim)
		_, back := compute(seq, backward, lim)
		_, two := compute(seq, twosided, lim)
		if lcslen(two) != lcslen(forw) || lcslen(forw) != lcslen(back) {
			t.Logf("\n%v\n%v\n%v", forw, back, two)
			t.Fatalf("%d forw:%d back:%d two:%d", i, lcslen(forw), lcslen(back), lcslen(two))
		}
		if !two.valid() || !forw.valid() || !back.valid() {
			t.Errorf("check failure")
		}
	}
}

// TestDiffAPI tests the public API functions (Diff{Bytes,Strings,Runes})
// to ensure at least miminal parity of the three representations.
func TestDiffAPI(t *testing.T) {
	for _, test := range []struct {
		a, b                              string
		wantStrings, wantBytes, wantRunes string
	}{
		{"abcXdef", "abcxdef", "[{3 4 3 4}]", "[{3 4 3 4}]", "[{3 4 3 4}]"}, // ASCII
		{"abcωdef", "abcΩdef", "[{3 5 3 5}]", "[{3 5 3 5}]", "[{3 4 3 4}]"}, // non-ASCII
	} {

		gotStrings := fmt.Sprint(DiffStrings(test.a, test.b))
		if gotStrings != test.wantStrings {
			t.Errorf("DiffStrings(%q, %q) = %v, want %v",
				test.a, test.b, gotStrings, test.wantStrings)
		}
		gotBytes := fmt.Sprint(DiffBytes([]byte(test.a), []byte(test.b)))
		if gotBytes != test.wantBytes {
			t.Errorf("DiffBytes(%q, %q) = %v, want %v",
				test.a, test.b, gotBytes, test.wantBytes)
		}
		gotRunes := fmt.Sprint(DiffRunes([]rune(test.a), []rune(test.b)))
		if gotRunes != test.wantRunes {
			t.Errorf("DiffRunes(%q, %q) = %v, want %v",
				test.a, test.b, gotRunes, test.wantRunes)
		}
	}
}

func BenchmarkTwoOld(b *testing.B) {
	tests := genBench("abc", 96)
	for i := 0; i < b.N; i++ {
		for _, tt := range tests {
			_, two := compute(stringSeqs{tt.before, tt.after}, twosided, 100)
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
			_, two := compute(stringSeqs{tt.before, tt.after}, forward, 100)
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

// This benchmark represents a common case for a diff command:
// large file with a single relatively small diff in the middle.
// (It's not clear whether this is representative of gopls workloads
// or whether it is important to gopls diff performance.)
//
// TODO(adonovan) opt: it could be much faster.  For example,
// comparing a file against itself is about 10x faster than with the
// small deletion in the middle. Strangely, comparing a file against
// itself minus the last byte is faster still; I don't know why.
// There is much low-hanging fruit here for further improvement.
func BenchmarkLargeFileSmallDiff(b *testing.B) {
	data, err := ioutil.ReadFile("old.go") // large file
	if err != nil {
		log.Fatal(err)
	}

	n := len(data)

	src := string(data)
	dst := src[:n*49/100] + src[n*51/100:] // remove 2% from the middle
	b.Run("string", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			compute(stringSeqs{src, dst}, twosided, len(src)+len(dst))
		}
	})

	srcBytes := []byte(src)
	dstBytes := []byte(dst)
	b.Run("bytes", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			compute(bytesSeqs{srcBytes, dstBytes}, twosided, len(srcBytes)+len(dstBytes))
		}
	})

	srcRunes := []rune(src)
	dstRunes := []rune(dst)
	b.Run("runes", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			compute(runesSeqs{srcRunes, dstRunes}, twosided, len(srcRunes)+len(dstRunes))
		}
	})
}
