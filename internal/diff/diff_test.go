// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package diff_test

import (
	"math/rand"
	"reflect"
	"testing"
	"unicode/utf8"

	"golang.org/x/tools/internal/diff"
	"golang.org/x/tools/internal/diff/difftest"
)

func TestApply(t *testing.T) {
	for _, tc := range difftest.TestCases {
		t.Run(tc.Name, func(t *testing.T) {
			got, err := diff.Apply(tc.In, tc.Edits)
			if err != nil {
				t.Fatalf("Apply(Edits) failed: %v", err)
			}
			if got != tc.Out {
				t.Errorf("Apply(Edits): got %q, want %q", got, tc.Out)
			}
			if tc.LineEdits != nil {
				got, err := diff.Apply(tc.In, tc.LineEdits)
				if err != nil {
					t.Fatalf("Apply(LineEdits) failed: %v", err)
				}
				if got != tc.Out {
					t.Errorf("Apply(LineEdits): got %q, want %q", got, tc.Out)
				}
			}
		})
	}
}

func TestNEdits(t *testing.T) {
	for _, tc := range difftest.TestCases {
		edits := diff.Strings(tc.In, tc.Out)
		got, err := diff.Apply(tc.In, edits)
		if err != nil {
			t.Fatalf("Apply failed: %v", err)
		}
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
		got, err := diff.Apply(a, edits)
		if err != nil {
			t.Fatalf("Apply failed: %v", err)
		}
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
		got, err := diff.Apply(a, edits)
		if err != nil {
			t.Fatalf("Apply failed: %v", err)
		}
		if got != b {
			t.Fatalf("applying diff(%q, %q) gives %q; edits=%v", a, b, got, edits)
		}
	})
}

func TestLineEdits(t *testing.T) {
	for _, tc := range difftest.TestCases {
		t.Run(tc.Name, func(t *testing.T) {
			// if line edits not specified, it is the same as edits
			edits := tc.LineEdits
			if edits == nil {
				edits = tc.Edits
			}
			got, err := diff.LineEdits(tc.In, tc.Edits)
			if err != nil {
				t.Fatalf("LineEdits: %v", err)
			}
			if !reflect.DeepEqual(got, edits) {
				t.Errorf("LineEdits got %q, want %q", got, edits)
			}
		})
	}
}

func TestToUnified(t *testing.T) {
	for _, tc := range difftest.TestCases {
		t.Run(tc.Name, func(t *testing.T) {
			unified, err := diff.ToUnified(difftest.FileA, difftest.FileB, tc.In, tc.Edits)
			if err != nil {
				t.Fatal(err)
			}
			if unified != tc.Unified {
				t.Errorf("Unified(Edits): got diff:\n%v\nexpected:\n%v", unified, tc.Unified)
			}
			if tc.LineEdits != nil {
				unified, err := diff.ToUnified(difftest.FileA, difftest.FileB, tc.In, tc.LineEdits)
				if err != nil {
					t.Fatal(err)
				}
				if unified != tc.Unified {
					t.Errorf("Unified(LineEdits): got diff:\n%v\nexpected:\n%v", unified, tc.Unified)
				}
			}
		})
	}
}

func TestRegressionOld001(t *testing.T) {
	a := "// Copyright 2019 The Go Authors. All rights reserved.\n// Use of this source code is governed by a BSD-style\n// license that can be found in the LICENSE file.\n\npackage diff_test\n\nimport (\n\t\"fmt\"\n\t\"math/rand\"\n\t\"strings\"\n\t\"testing\"\n\n\t\"golang.org/x/tools/gopls/internal/lsp/diff\"\n\t\"golang.org/x/tools/internal/diff/difftest\"\n\t\"golang.org/x/tools/gopls/internal/span\"\n)\n"

	b := "// Copyright 2019 The Go Authors. All rights reserved.\n// Use of this source code is governed by a BSD-style\n// license that can be found in the LICENSE file.\n\npackage diff_test\n\nimport (\n\t\"fmt\"\n\t\"math/rand\"\n\t\"strings\"\n\t\"testing\"\n\n\t\"github.com/google/safehtml/template\"\n\t\"golang.org/x/tools/gopls/internal/lsp/diff\"\n\t\"golang.org/x/tools/internal/diff/difftest\"\n\t\"golang.org/x/tools/gopls/internal/span\"\n)\n"
	diffs := diff.Strings(a, b)
	got, err := diff.Apply(a, diffs)
	if err != nil {
		t.Fatalf("Apply failed: %v", err)
	}
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
	got, err := diff.Apply(a, diffs)
	if err != nil {
		t.Fatalf("Apply failed: %v", err)
	}
	if got != b {
		i := 0
		for ; i < len(a) && i < len(b) && got[i] == b[i]; i++ {
		}
		t.Errorf("oops %vd\n%q\n%q", diffs, got, b)
		t.Errorf("\n%q\n%q", got[i:], b[i:])
	}
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
