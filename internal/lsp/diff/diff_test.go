// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package diff_test

import (
	"fmt"
	"math/rand"
	"strings"
	"testing"

	"golang.org/x/tools/internal/lsp/diff"
	"golang.org/x/tools/internal/lsp/diff/difftest"
	"golang.org/x/tools/internal/span"
)

func TestApplyEdits(t *testing.T) {
	for _, tc := range difftest.TestCases {
		t.Run(tc.Name, func(t *testing.T) {
			t.Helper()
			if got := diff.ApplyEdits(tc.In, tc.Edits); got != tc.Out {
				t.Errorf("ApplyEdits edits got %q, want %q", got, tc.Out)
			}
			if tc.LineEdits != nil {
				if got := diff.ApplyEdits(tc.In, tc.LineEdits); got != tc.Out {
					t.Errorf("ApplyEdits lineEdits got %q, want %q", got, tc.Out)
				}
			}
		})
	}
}

func TestNEdits(t *testing.T) {
	for i, tc := range difftest.TestCases {
		sp := fmt.Sprintf("file://%s.%d", tc.Name, i)
		edits, err := diff.NComputeEdits(span.URI(sp), tc.In, tc.Out)
		if err != nil {
			t.Fatal(err)
		}
		got := diff.ApplyEdits(tc.In, edits)
		if got != tc.Out {
			t.Fatalf("%s: got %q wanted %q", tc.Name, got, tc.Out)
		}
		if len(edits) < len(tc.Edits) { // should find subline edits
			t.Errorf("got %v, expected %v for %#v", edits, tc.Edits, tc)
		}
	}
}

func TestNRandom(t *testing.T) {
	rand.Seed(1)
	for i := 0; i < 1000; i++ {
		fname := fmt.Sprintf("file://%x", i)
		a := randstr("abω", 16)
		b := randstr("abωc", 16)
		edits, err := diff.NComputeEdits(span.URI(fname), a, b)
		if err != nil {
			t.Fatalf("%q,%q %v", a, b, err)
		}
		got := diff.ApplyEdits(a, edits)
		if got != b {
			t.Fatalf("%d: got %q, wanted %q, starting with %q", i, got, b, a)
		}
	}
}

func TestNLinesRandom(t *testing.T) {
	rand.Seed(2)
	for i := 0; i < 1000; i++ {
		fname := fmt.Sprintf("file://%x", i)
		x := randlines("abω", 4) // avg line length is 6, want a change every 3rd line or so
		v := []rune(x)
		for i := 0; i < len(v); i++ {
			if rand.Float64() < .05 {
				v[i] = 'N'
			}
		}
		y := string(v)
		// occasionally remove the trailing \n
		if rand.Float64() < .1 {
			x = x[:len(x)-1]
		}
		if rand.Float64() < .1 {
			y = y[:len(y)-1]
		}
		a, b := strings.SplitAfter(x, "\n"), strings.SplitAfter(y, "\n")
		edits, err := diff.NComputeLineEdits(span.URI(fname), a, b)
		if err != nil {
			t.Fatalf("%q,%q %v", a, b, err)
		}
		got := diff.ApplyEdits(x, edits)
		if got != y {
			t.Fatalf("%d: got\n%q, wanted\n%q, starting with %q", i, got, y, a)
		}
	}
}

func TestLineEdits(t *testing.T) {
	for _, tc := range difftest.TestCases {
		t.Run(tc.Name, func(t *testing.T) {
			t.Helper()
			// if line edits not specified, it is the same as edits
			edits := tc.LineEdits
			if edits == nil {
				edits = tc.Edits
			}
			if got := diff.LineEdits(tc.In, tc.Edits); diffEdits(got, edits) {
				t.Errorf("LineEdits got %q, want %q", got, edits)
			}
		})
	}
}

func TestUnified(t *testing.T) {
	for _, tc := range difftest.TestCases {
		t.Run(tc.Name, func(t *testing.T) {
			t.Helper()
			unified := fmt.Sprint(diff.ToUnified(difftest.FileA, difftest.FileB, tc.In, tc.Edits))
			if unified != tc.Unified {
				t.Errorf("edits got diff:\n%v\nexpected:\n%v", unified, tc.Unified)
			}
			if tc.LineEdits != nil {
				unified := fmt.Sprint(diff.ToUnified(difftest.FileA, difftest.FileB, tc.In, tc.LineEdits))
				if unified != tc.Unified {
					t.Errorf("lineEdits got diff:\n%v\nexpected:\n%v", unified, tc.Unified)
				}
			}
		})
	}
}

func TestRegressionOld001(t *testing.T) {
	a := "// Copyright 2019 The Go Authors. All rights reserved.\n// Use of this source code is governed by a BSD-style\n// license that can be found in the LICENSE file.\n\npackage diff_test\n\nimport (\n\t\"fmt\"\n\t\"math/rand\"\n\t\"strings\"\n\t\"testing\"\n\n\t\"golang.org/x/tools/internal/lsp/diff\"\n\t\"golang.org/x/tools/internal/lsp/diff/difftest\"\n\t\"golang.org/x/tools/internal/span\"\n)\n"

	b := "// Copyright 2019 The Go Authors. All rights reserved.\n// Use of this source code is governed by a BSD-style\n// license that can be found in the LICENSE file.\n\npackage diff_test\n\nimport (\n\t\"fmt\"\n\t\"math/rand\"\n\t\"strings\"\n\t\"testing\"\n\n\t\"github.com/google/safehtml/template\"\n\t\"golang.org/x/tools/internal/lsp/diff\"\n\t\"golang.org/x/tools/internal/lsp/diff/difftest\"\n\t\"golang.org/x/tools/internal/span\"\n)\n"
	diffs, err := diff.NComputeEdits(span.URI("file://one"), a, b)
	if err != nil {
		t.Error(err)
	}
	got := diff.ApplyEdits(a, diffs)
	if got != b {
		i := 0
		for ; i < len(a) && i < len(b) && got[i] == b[i]; i++ {
		}
		t.Errorf("oops %vd\n%q\n%q", diffs, got, b)
		t.Errorf("\n%q\n%q", got[i:], b[i:])
	}
}

func TestRegressionOld002(t *testing.T) {
	a := "n\"\n)\n"
	b := "n\"\n\t\"golang.org/x//nnal/stack\"\n)\n"
	diffs, err := diff.NComputeEdits(span.URI("file://two"), a, b)
	if err != nil {
		t.Error(err)
	}
	got := diff.ApplyEdits(a, diffs)
	if got != b {
		i := 0
		for ; i < len(a) && i < len(b) && got[i] == b[i]; i++ {
		}
		t.Errorf("oops %vd\n%q\n%q", diffs, got, b)
		t.Errorf("\n%q\n%q", got[i:], b[i:])
	}
}

func diffEdits(got, want []diff.TextEdit) bool {
	if len(got) != len(want) {
		return true
	}
	for i, w := range want {
		g := got[i]
		if span.Compare(w.Span, g.Span) != 0 {
			return true
		}
		if w.NewText != g.NewText {
			return true
		}
	}
	return false
}

// return a random string of length n made of characters from s
func randstr(s string, n int) string {
	src := []rune(s)
	x := make([]rune, n)
	for i := 0; i < n; i++ {
		x[i] = src[rand.Intn(len(src))]
	}
	return string(x)
}

// return some random lines, all ending with \n
func randlines(s string, n int) string {
	src := []rune(s)
	var b strings.Builder
	for i := 0; i < n; i++ {
		for j := 0; j < 4+rand.Intn(4); j++ {
			b.WriteRune(src[rand.Intn(len(src))])
		}
		b.WriteByte('\n')
	}
	return b.String()
}
