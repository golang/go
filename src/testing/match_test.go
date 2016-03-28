// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testing

import (
	"regexp"
	"unicode"
)

// Verify that our IsSpace agrees with unicode.IsSpace.
func TestIsSpace(t *T) {
	n := 0
	for r := rune(0); r <= unicode.MaxRune; r++ {
		if isSpace(r) != unicode.IsSpace(r) {
			t.Errorf("IsSpace(%U)=%t incorrect", r, isSpace(r))
			n++
			if n > 10 {
				return
			}
		}
	}
}

func TestNaming(t *T) {
	m := newMatcher(regexp.MatchString, "", "")

	parent := &common{name: "x", level: 1} // top-level test.

	// Rig the matcher with some preloaded values.
	m.subNames["x/b"] = 1000

	testCases := []struct {
		name, want string
	}{
		// Uniqueness
		{"", "x/#00"},
		{"", "x/#01"},

		{"t", "x/t"},
		{"t", "x/t#01"},
		{"t", "x/t#02"},

		{"a#01", "x/a#01"}, // user has subtest with this name.
		{"a", "x/a"},       // doesn't conflict with this name.
		{"a", "x/a#01#01"}, // conflict, add disambiguating string.
		{"a", "x/a#02"},    // This string is claimed now, so resume
		{"a", "x/a#03"},    // with counting.
		{"a#02", "x/a#02#01"},

		{"b", "x/b#1000"}, // rigged, see above
		{"b", "x/b#1001"},

		// // Sanitizing
		{"A:1 B:2", "x/A:1_B:2"},
		{"s\t\r\u00a0", "x/s___"},
		{"\x01", `x/\x01`},
		{"\U0010ffff", `x/\U0010ffff`},
	}

	for i, tc := range testCases {
		if got, _ := m.fullName(parent, tc.name); got != tc.want {
			t.Errorf("%d:%s: got %q; want %q", i, tc.name, got, tc.want)
		}
	}
}
