// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package difftest supplies a set of tests that will operate on any
// implementation of a diff algorithm as exposed by
// "golang.org/x/tools/internal/lsp/diff"
package difftest

import (
	"fmt"
	"testing"

	"golang.org/x/tools/internal/lsp/diff"
	"golang.org/x/tools/internal/span"
)

const (
	FileA         = "from"
	FileB         = "to"
	UnifiedPrefix = "--- " + FileA + "\n+++ " + FileB + "\n"
)

var TestCases = []struct {
	Name, In, Out, Unified string
	Edits, LineEdits       []diff.TextEdit
	NoDiff                 bool
}{{
	Name: "empty",
	In:   "",
	Out:  "",
}, {
	Name: "no_diff",
	In:   "gargantuan\n",
	Out:  "gargantuan\n",
}, {
	Name: "replace_all",
	In:   "fruit\n",
	Out:  "cheese\n",
	Unified: UnifiedPrefix + `
@@ -1 +1 @@
-fruit
+cheese
`[1:],
	Edits:     []diff.TextEdit{{Span: newSpan(0, 5), NewText: "cheese"}},
	LineEdits: []diff.TextEdit{{Span: newSpan(0, 6), NewText: "cheese\n"}},
}, {
	Name: "insert_rune",
	In:   "gord\n",
	Out:  "gourd\n",
	Unified: UnifiedPrefix + `
@@ -1 +1 @@
-gord
+gourd
`[1:],
	Edits:     []diff.TextEdit{{Span: newSpan(2, 2), NewText: "u"}},
	LineEdits: []diff.TextEdit{{Span: newSpan(0, 5), NewText: "gourd\n"}},
}, {
	Name: "delete_rune",
	In:   "groat\n",
	Out:  "goat\n",
	Unified: UnifiedPrefix + `
@@ -1 +1 @@
-groat
+goat
`[1:],
	Edits:     []diff.TextEdit{{Span: newSpan(1, 2), NewText: ""}},
	LineEdits: []diff.TextEdit{{Span: newSpan(0, 6), NewText: "goat\n"}},
}, {
	Name: "replace_rune",
	In:   "loud\n",
	Out:  "lord\n",
	Unified: UnifiedPrefix + `
@@ -1 +1 @@
-loud
+lord
`[1:],
	Edits:     []diff.TextEdit{{Span: newSpan(2, 3), NewText: "r"}},
	LineEdits: []diff.TextEdit{{Span: newSpan(0, 5), NewText: "lord\n"}},
}, {
	Name: "replace_partials",
	In:   "blanket\n",
	Out:  "bunker\n",
	Unified: UnifiedPrefix + `
@@ -1 +1 @@
-blanket
+bunker
`[1:],
	Edits: []diff.TextEdit{
		{Span: newSpan(1, 3), NewText: "u"},
		{Span: newSpan(6, 7), NewText: "r"},
	},
	LineEdits: []diff.TextEdit{{Span: newSpan(0, 8), NewText: "bunker\n"}},
}, {
	Name: "insert_line",
	In:   "1: one\n3: three\n",
	Out:  "1: one\n2: two\n3: three\n",
	Unified: UnifiedPrefix + `
@@ -1,2 +1,3 @@
 1: one
+2: two
 3: three
`[1:],
	Edits: []diff.TextEdit{{Span: newSpan(7, 7), NewText: "2: two\n"}},
}, {
	Name: "replace_no_newline",
	In:   "A",
	Out:  "B",
	Unified: UnifiedPrefix + `
@@ -1 +1 @@
-A
\ No newline at end of file
+B
\ No newline at end of file
`[1:],
	Edits: []diff.TextEdit{{Span: newSpan(0, 1), NewText: "B"}},
}, {
	Name: "add_end",
	In:   "A",
	Out:  "AB",
	Unified: UnifiedPrefix + `
@@ -1 +1 @@
-A
\ No newline at end of file
+AB
\ No newline at end of file
`[1:],
	Edits:     []diff.TextEdit{{Span: newSpan(1, 1), NewText: "B"}},
	LineEdits: []diff.TextEdit{{Span: newSpan(0, 1), NewText: "AB"}},
}, {
	Name: "add_newline",
	In:   "A",
	Out:  "A\n",
	Unified: UnifiedPrefix + `
@@ -1 +1 @@
-A
\ No newline at end of file
+A
`[1:],
	Edits:     []diff.TextEdit{{Span: newSpan(1, 1), NewText: "\n"}},
	LineEdits: []diff.TextEdit{{Span: newSpan(0, 1), NewText: "A\n"}},
}, {
	Name: "delete_front",
	In:   "A\nB\nC\nA\nB\nB\nA\n",
	Out:  "C\nB\nA\nB\nA\nC\n",
	Unified: UnifiedPrefix + `
@@ -1,7 +1,6 @@
-A
-B
 C
+B
 A
 B
-B
 A
+C
`[1:],
	Edits: []diff.TextEdit{
		{Span: newSpan(0, 4), NewText: ""},
		{Span: newSpan(6, 6), NewText: "B\n"},
		{Span: newSpan(10, 12), NewText: ""},
		{Span: newSpan(14, 14), NewText: "C\n"},
	},
	NoDiff: true, // diff algorithm produces different delete/insert pattern
},
	{
		Name: "replace_last_line",
		In:   "A\nB\n",
		Out:  "A\nC\n\n",
		Unified: UnifiedPrefix + `
@@ -1,2 +1,3 @@
 A
-B
+C
+
`[1:],
		Edits:     []diff.TextEdit{{Span: newSpan(2, 3), NewText: "C\n"}},
		LineEdits: []diff.TextEdit{{Span: newSpan(2, 4), NewText: "C\n\n"}},
	},
	{
		Name: "mulitple_replace",
		In:   "A\nB\nC\nD\nE\nF\nG\n",
		Out:  "A\nH\nI\nJ\nE\nF\nK\n",
		Unified: UnifiedPrefix + `
@@ -1,7 +1,7 @@
 A
-B
-C
-D
+H
+I
+J
 E
 F
-G
+K
`[1:],
		Edits: []diff.TextEdit{
			{Span: newSpan(2, 8), NewText: "H\nI\nJ\n"},
			{Span: newSpan(12, 14), NewText: "K\n"},
		},
		NoDiff: true, // diff algorithm produces different delete/insert pattern
	},
}

func init() {
	// expand all the spans to full versions
	// we need them all to have their line number and column
	for _, tc := range TestCases {
		c := span.NewContentConverter("", []byte(tc.In))
		for i := range tc.Edits {
			tc.Edits[i].Span, _ = tc.Edits[i].Span.WithAll(c)
		}
		for i := range tc.LineEdits {
			tc.LineEdits[i].Span, _ = tc.LineEdits[i].Span.WithAll(c)
		}
	}
}

func DiffTest(t *testing.T, compute diff.ComputeEdits) {
	t.Helper()
	for _, test := range TestCases {
		t.Run(test.Name, func(t *testing.T) {
			t.Helper()
			edits := compute(span.FileURI("/"+test.Name), test.In, test.Out)
			got := diff.ApplyEdits(test.In, edits)
			unified := fmt.Sprint(diff.ToUnified(FileA, FileB, test.In, edits))
			if got != test.Out {
				t.Errorf("got patched:\n%v\nfrom diff:\n%v\nexpected:\n%v", got, unified, test.Out)
			}
			if !test.NoDiff && unified != test.Unified {
				t.Errorf("got diff:\n%v\nexpected:\n%v", unified, test.Unified)
			}
		})
	}
}

func newSpan(start, end int) span.Span {
	return span.New("", span.NewPoint(0, 0, start), span.NewPoint(0, 0, end))
}
