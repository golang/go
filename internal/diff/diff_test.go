// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package diff_test

import (
	"math/rand"
	"reflect"
	"strings"
	"testing"
	"unicode/utf8"

	"golang.org/x/tools/internal/diff"
	"golang.org/x/tools/internal/diff/difftest"
)

func TestApply(t *testing.T) {
	for _, tc := range difftest.TestCases {
		t.Run(tc.Name, func(t *testing.T) {
			if got := diff.Apply(tc.In, tc.Edits); got != tc.Out {
				t.Errorf("Apply(Edits): got %q, want %q", got, tc.Out)
			}
			if tc.LineEdits != nil {
				if got := diff.Apply(tc.In, tc.LineEdits); got != tc.Out {
					t.Errorf("Apply(LineEdits): got %q, want %q", got, tc.Out)
				}
			}
		})
	}
}

func TestNEdits(t *testing.T) {
	for _, tc := range difftest.TestCases {
		edits := diff.Strings(tc.In, tc.Out)
		got := diff.Apply(tc.In, edits)
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
		a := randstr("abω", 16)
		b := randstr("abωc", 16)
		edits := diff.Strings(a, b)
		got := diff.Apply(a, edits)
		if got != b {
			t.Fatalf("%d: got %q, wanted %q, starting with %q", i, got, b, a)
		}
	}
}

// $ go test -fuzz=FuzzRoundTrip ./internal/diff
func FuzzRoundTrip(f *testing.F) {
	f.Fuzz(func(t *testing.T, a, b string) {
		if !utf8.ValidString(a) || !utf8.ValidString(b) {
			return // inputs must be text
		}
		edits := diff.Strings(a, b)
		got := diff.Apply(a, edits)
		if got != b {
			t.Fatalf("applying diff(%q, %q) gives %q; edits=%v", a, b, got, edits)
		}
	})
}

func TestNLinesRandom(t *testing.T) {
	rand.Seed(2)
	for i := 0; i < 1000; i++ {
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
		edits := diff.Lines(a, b)
		got := diff.Apply(x, edits)
		if got != y {
			t.Fatalf("%d: got\n%q, wanted\n%q, starting with %q", i, got, y, a)
		}
	}
}

func TestLineEdits(t *testing.T) {
	for _, tc := range difftest.TestCases {
		t.Run(tc.Name, func(t *testing.T) {
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
			unified := diff.Unified(difftest.FileA, difftest.FileB, tc.In, tc.Edits)
			if unified != tc.Unified {
				t.Errorf("Unified(Edits): got diff:\n%v\nexpected:\n%v", unified, tc.Unified)
			}
			if tc.LineEdits != nil {
				unified := diff.Unified(difftest.FileA, difftest.FileB, tc.In, tc.LineEdits)
				if unified != tc.Unified {
					t.Errorf("Unified(LineEdits): got diff:\n%v\nexpected:\n%v", unified, tc.Unified)
				}
			}
		})
	}
}

func TestRegressionOld001(t *testing.T) {
	a := "// Copyright 2019 The Go Authors. All rights reserved.\n// Use of this source code is governed by a BSD-style\n// license that can be found in the LICENSE file.\n\npackage diff_test\n\nimport (\n\t\"fmt\"\n\t\"math/rand\"\n\t\"strings\"\n\t\"testing\"\n\n\t\"golang.org/x/tools/gopls/internal/lsp/diff\"\n\t\"golang.org/x/tools/internal/diff/difftest\"\n\t\"golang.org/x/tools/internal/span\"\n)\n"

	b := "// Copyright 2019 The Go Authors. All rights reserved.\n// Use of this source code is governed by a BSD-style\n// license that can be found in the LICENSE file.\n\npackage diff_test\n\nimport (\n\t\"fmt\"\n\t\"math/rand\"\n\t\"strings\"\n\t\"testing\"\n\n\t\"github.com/google/safehtml/template\"\n\t\"golang.org/x/tools/gopls/internal/lsp/diff\"\n\t\"golang.org/x/tools/internal/diff/difftest\"\n\t\"golang.org/x/tools/internal/span\"\n)\n"
	diffs := diff.Strings(a, b)
	got := diff.Apply(a, diffs)
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
	diffs := diff.Strings(a, b)
	got := diff.Apply(a, diffs)
	if got != b {
		i := 0
		for ; i < len(a) && i < len(b) && got[i] == b[i]; i++ {
		}
		t.Errorf("oops %vd\n%q\n%q", diffs, got, b)
		t.Errorf("\n%q\n%q", got[i:], b[i:])
	}
}

func diffEdits(got, want []diff.Edit) bool {
	return !reflect.DeepEqual(got, want)
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
