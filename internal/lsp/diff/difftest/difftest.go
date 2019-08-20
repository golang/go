// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package difftest supplies a set of tests that will operate on any
// implementation of a diff algorithm as exposed by
// "golang.org/x/tools/internal/lsp/diff"
package difftest

import (
	"testing"

	"golang.org/x/tools/internal/lsp/diff"
	"golang.org/x/tools/internal/span"
)

func DiffTest(t *testing.T) {
	t.Helper()
	for _, test := range []struct{ name, in, out, unified string }{{
		name: "empty",
		in:   "",
		out:  "",
	}, {
		name: "no_diff",
		in:   "gargantuan",
		out:  "gargantuan",
	}, {
		name: "insert_rune",
		in:   "gord",
		out:  "gourd",
	}, {
		name: "delete_rune",
		in:   "groat",
		out:  "goat",
	}, {
		name: "replace_rune",
		in:   "loud",
		out:  "lord",
	}, {
		name: "insert_line",
		in:   "one\nthree\n",
		out:  "one\ntwo\nthree\n",
	}} {
		edits := diff.ComputeEdits(span.FileURI("/"+test.name), test.in, test.out)
		got := diff.ApplyEdits(test.in, edits)
		if got != test.out {
			t.Logf("test %v had diff:%v\n", test.name, diff.ToUnified(test.name+".orig", test.name, test.in, edits))
			t.Errorf("diff %v got:\n%v\nexpected:\n%v", test.name, got, test.out)
		}
	}
}
