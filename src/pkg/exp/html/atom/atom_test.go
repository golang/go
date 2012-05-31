// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package atom

import (
	"testing"
)

func TestHits(t *testing.T) {
	for i, s := range table {
		got := Lookup([]byte(s))
		if got != Atom(i) {
			t.Errorf("Lookup(%q): got %d, want %d", s, got, i)
		}
	}
}

func TestMisses(t *testing.T) {
	testCases := []string{
		"",
		"\x00",
		"\xff",
		"A",
		"DIV",
		"Div",
		"dIV",
		"aa",
		"a\x00",
		"ab",
		"abb",
		"abbr0",
		"abbr ",
		" abbr",
		" a",
		"acceptcharset",
		"acceptCharset",
		"accept_charset",
		"h0",
		"h1h2",
		"h7",
		"onClick",
		"λ",
	}
	for _, tc := range testCases {
		got := Lookup([]byte(tc))
		if got != 0 {
			t.Errorf("Lookup(%q): got %d, want 0", tc, got)
		}
	}
}
