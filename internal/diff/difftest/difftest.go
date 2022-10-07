// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package difftest supplies a set of tests that will operate on any
// implementation of a diff algorithm as exposed by
// "golang.org/x/tools/internal/diff"
package difftest

import (
	"testing"

	"golang.org/x/tools/internal/diff"
)

const (
	FileA         = "from"
	FileB         = "to"
	UnifiedPrefix = "--- " + FileA + "\n+++ " + FileB + "\n"
)

var TestCases = []struct {
	Name, In, Out, Unified string
	Edits, LineEdits       []diff.Edit
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
	Edits:     []diff.Edit{{Start: 0, End: 5, New: "cheese"}},
	LineEdits: []diff.Edit{{Start: 0, End: 6, New: "cheese\n"}},
}, {
	Name: "insert_rune",
	In:   "gord\n",
	Out:  "gourd\n",
	Unified: UnifiedPrefix + `
@@ -1 +1 @@
-gord
+gourd
`[1:],
	Edits:     []diff.Edit{{Start: 2, End: 2, New: "u"}},
	LineEdits: []diff.Edit{{Start: 0, End: 5, New: "gourd\n"}},
}, {
	Name: "delete_rune",
	In:   "groat\n",
	Out:  "goat\n",
	Unified: UnifiedPrefix + `
@@ -1 +1 @@
-groat
+goat
`[1:],
	Edits:     []diff.Edit{{Start: 1, End: 2, New: ""}},
	LineEdits: []diff.Edit{{Start: 0, End: 6, New: "goat\n"}},
}, {
	Name: "replace_rune",
	In:   "loud\n",
	Out:  "lord\n",
	Unified: UnifiedPrefix + `
@@ -1 +1 @@
-loud
+lord
`[1:],
	Edits:     []diff.Edit{{Start: 2, End: 3, New: "r"}},
	LineEdits: []diff.Edit{{Start: 0, End: 5, New: "lord\n"}},
}, {
	Name: "replace_partials",
	In:   "blanket\n",
	Out:  "bunker\n",
	Unified: UnifiedPrefix + `
@@ -1 +1 @@
-blanket
+bunker
`[1:],
	Edits: []diff.Edit{
		{Start: 1, End: 3, New: "u"},
		{Start: 6, End: 7, New: "r"},
	},
	LineEdits: []diff.Edit{{Start: 0, End: 8, New: "bunker\n"}},
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
	Edits: []diff.Edit{{Start: 7, End: 7, New: "2: two\n"}},
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
	Edits: []diff.Edit{{Start: 0, End: 1, New: "B"}},
}, {
	Name: "append_empty",
	In:   "", // GNU diff -u special case: -0,0
	Out:  "AB\nC",
	Unified: UnifiedPrefix + `
@@ -0,0 +1,2 @@
+AB
+C
\ No newline at end of file
`[1:],
	Edits:     []diff.Edit{{Start: 0, End: 0, New: "AB\nC"}},
	LineEdits: []diff.Edit{{Start: 0, End: 0, New: "AB\nC"}},
},
	// TODO(adonovan): fix this test: GNU diff -u prints "+1,2", Unifies prints "+1,3".
	// 	{
	// 		Name: "add_start",
	// 		In:   "A",
	// 		Out:  "B\nCA",
	// 		Unified: UnifiedPrefix + `
	// @@ -1 +1,2 @@
	// -A
	// \ No newline at end of file
	// +B
	// +CA
	// \ No newline at end of file
	// `[1:],
	// 		Edits:     []diff.TextEdit{{Span: newSpan(0, 0), NewText: "B\nC"}},
	// 		LineEdits: []diff.TextEdit{{Span: newSpan(0, 0), NewText: "B\nC"}},
	// 	},
	{
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
		Edits:     []diff.Edit{{Start: 1, End: 1, New: "B"}},
		LineEdits: []diff.Edit{{Start: 0, End: 1, New: "AB"}},
	}, {
		Name: "add_empty",
		In:   "",
		Out:  "AB\nC",
		Unified: UnifiedPrefix + `
@@ -0,0 +1,2 @@
+AB
+C
\ No newline at end of file
`[1:],
		Edits:     []diff.Edit{{Start: 0, End: 0, New: "AB\nC"}},
		LineEdits: []diff.Edit{{Start: 0, End: 0, New: "AB\nC"}},
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
		Edits:     []diff.Edit{{Start: 1, End: 1, New: "\n"}},
		LineEdits: []diff.Edit{{Start: 0, End: 1, New: "A\n"}},
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
		NoDiff: true, // unified diff is different but valid
		Edits: []diff.Edit{
			{Start: 0, End: 4, New: ""},
			{Start: 6, End: 6, New: "B\n"},
			{Start: 10, End: 12, New: ""},
			{Start: 14, End: 14, New: "C\n"},
		},
	}, {
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
		Edits:     []diff.Edit{{Start: 2, End: 3, New: "C\n"}},
		LineEdits: []diff.Edit{{Start: 2, End: 4, New: "C\n\n"}},
	},
	{
		Name: "multiple_replace",
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
		Edits: []diff.Edit{
			{Start: 2, End: 8, New: "H\nI\nJ\n"},
			{Start: 12, End: 14, New: "K\n"},
		},
		NoDiff: true, // diff algorithm produces different delete/insert pattern
	},
}

func DiffTest(t *testing.T, compute func(before, after string) []diff.Edit) {
	for _, test := range TestCases {
		t.Run(test.Name, func(t *testing.T) {
			edits := compute(test.In, test.Out)
			got, err := diff.Apply(test.In, edits)
			if err != nil {
				t.Fatalf("Apply failed: %v", err)
			}
			unified, err := diff.ToUnified(FileA, FileB, test.In, edits)
			if err != nil {
				t.Fatalf("ToUnified: %v", err)
			}
			if got != test.Out {
				t.Errorf("Apply: got patched:\n%v\nfrom diff:\n%v\nexpected:\n%v", got, unified, test.Out)
			}
			if !test.NoDiff && unified != test.Unified {
				t.Errorf("Unified: got diff:\n%v\nexpected:\n%v", unified, test.Unified)
			}
		})
	}
}
