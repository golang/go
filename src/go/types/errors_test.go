// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import "testing"

func TestStripSubscripts(t *testing.T) {
	for _, test := range []struct {
		in, want string
	}{
		{"", ""},
		{"   ", "   "},
		{"foo", "foo"},
		{"foo₀", "foo"},
		{"foo(T₀)", "foo(T)"},
	} {
		got := stripSubscripts(test.in)
		if got != test.want {
			t.Errorf("%q: got %q; want %q", test.in, got, test.want)
		}
	}
}
