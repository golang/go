// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import "testing"

func TestError(t *testing.T) {
	var err error_
	want := "no error"
	if got := err.msg(); got != want {
		t.Errorf("empty error: got %q, want %q", got, want)
	}

	want = "foo 42"
	err.addf(nopos, "foo %d", 42)
	if got := err.msg(); got != want {
		t.Errorf("simple error: got %q, want %q", got, want)
	}

	want = "foo 42\n\tbar 43"
	err.addf(nopos, "bar %d", 43)
	if got := err.msg(); got != want {
		t.Errorf("simple error: got %q, want %q", got, want)
	}
}

func TestStripAnnotations(t *testing.T) {
	for _, test := range []struct {
		in, want string
	}{
		{"", ""},
		{"   ", "   "},
		{"foo", "foo"},
		{"foo₀", "foo"},
		{"foo(T₀)", "foo(T)"},
	} {
		got := stripAnnotations(test.in)
		if got != test.want {
			t.Errorf("%q: got %q; want %q", test.in, got, test.want)
		}
	}
}
