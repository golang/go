// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unify

import (
	"strings"
	"testing"
)

func TestValueFilter(t *testing.T) {
	read := func(s string) *Value {
		t.Helper()
		c, err := Read(strings.NewReader(s), "<test>", ReadOpts{})
		if err != nil {
			t.Fatal(err)
		}
		return c.val
	}

	for _, test := range []struct {
		probe, v string
		want     bool
	}{
		{"{a: x}", "{a: x}", true},
		{"{a: x}", "{a: y}", false},
		{"{a: x}", `{a: "x|y"}`, true},
		{"{a: x}", `{a: "y|z"}`, false},
		{`{a: "y|z"}`, "{a: x}", false},
		// Unify doesn't check whether regexps intersect.
		{`{a: "w|x"}`, `{a: "y|z"}`, true},
		// Fields only on one side don't matter.
		{"{a: x, b: y}", "{b: z, c: x}", false},
		{"{a: x}", "{b: y}", true},
		// Top matches anything.
		{"{a: _, b: x}", "{a: {c: d}, b: y}", false},
		{"{a: x, b: y}", "{a: _, b: z}", false},
		// Unifying a with a non-String could bind variables or fail with an
		// error, so the filter can't predict anything about b.
		{"{a: x, b: y}", "{a: {c: d}, b: z}", true},
		{"{a: $v, b: y}", "{a: x, b: z}", true},
		{"{a: [x], b: y}", "{a: [x], b: z}", true},
		// But it can reject based on earlier fields.
		{"{a: x, b: {c: d}}", "{a: y, b: z}", false},
		// The filter only applies to Defs.
		{"{a: x}", "x", true},
		{"{a: x}", "$v", true},
	} {
		f := newValueFilter(read(test.probe))
		if got := f.mayUnify(read(test.v)); got != test.want {
			t.Errorf("filter %s: mayUnify(%s) = %v, want %v", test.probe, test.v, got, test.want)
		}
	}
}

// TestValueFilterError checks that filtering doesn't hide errors from fields
// that unification visits before a mismatched field.
func TestValueFilterError(t *testing.T) {
	read := func(s string) Closure {
		t.Helper()
		c, err := Read(strings.NewReader(s), "<test>", ReadOpts{})
		if err != nil {
			t.Fatal(err)
		}
		return c
	}
	_, err := Unify(read("!sum [{a: x, b: y}, {a: {c: d}, b: z}]"), read("{a: x, b: y}"))
	if err == nil || !strings.Contains(err.Error(), "kind mismatch") {
		t.Errorf("want kind mismatch error, got %v", err)
	}
}
