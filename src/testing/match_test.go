// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testing

import (
	"reflect"
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

func TestSplitRegexp(t *T) {
	res := func(s ...string) []string { return s }
	testCases := []struct {
		pattern string
		result  []string
	}{
		// Correct patterns
		// If a regexp pattern is correct, all split regexps need to be correct
		// as well.
		{"", res("")},
		{"/", res("", "")},
		{"//", res("", "", "")},
		{"A", res("A")},
		{"A/B", res("A", "B")},
		{"A/B/", res("A", "B", "")},
		{"/A/B/", res("", "A", "B", "")},
		{"[A]/(B)", res("[A]", "(B)")},
		{"[/]/[/]", res("[/]", "[/]")},
		{"[/]/[:/]", res("[/]", "[:/]")},
		{"/]", res("", "]")},
		{"]/", res("]", "")},
		{"]/[/]", res("]", "[/]")},
		{`([)/][(])`, res(`([)/][(])`)},
		{"[(]/[)]", res("[(]", "[)]")},

		// Faulty patterns
		// Errors in original should produce at least one faulty regexp in results.
		{")/", res(")/")},
		{")/(/)", res(")/(", ")")},
		{"a[/)b", res("a[/)b")},
		{"(/]", res("(/]")},
		{"(/", res("(/")},
		{"[/]/[/", res("[/]", "[/")},
		{`\p{/}`, res(`\p{`, "}")},
		{`\p/`, res(`\p`, "")},
		{`[[:/:]]`, res(`[[:/:]]`)},
	}
	for _, tc := range testCases {
		a := splitRegexp(tc.pattern)
		if !reflect.DeepEqual(a, tc.result) {
			t.Errorf("splitRegexp(%q) = %#v; want %#v", tc.pattern, a, tc.result)
		}

		// If there is any error in the pattern, one of the returned subpatterns
		// needs to have an error as well.
		if _, err := regexp.Compile(tc.pattern); err != nil {
			ok := true
			for _, re := range a {
				if _, err := regexp.Compile(re); err != nil {
					ok = false
				}
			}
			if ok {
				t.Errorf("%s: expected error in any of %q", tc.pattern, a)
			}
		}
	}
}

func TestMatcher(t *T) {
	testCases := []struct {
		pattern     string
		parent, sub string
		ok          bool
		partial     bool
	}{
		// Behavior without subtests.
		{"", "", "TestFoo", true, false},
		{"TestFoo", "", "TestFoo", true, false},
		{"TestFoo/", "", "TestFoo", true, true},
		{"TestFoo/bar/baz", "", "TestFoo", true, true},
		{"TestFoo", "", "TestBar", false, false},
		{"TestFoo/", "", "TestBar", false, false},
		{"TestFoo/bar/baz", "", "TestBar/bar/baz", false, false},

		// with subtests
		{"", "TestFoo", "x", true, false},
		{"TestFoo", "TestFoo", "x", true, false},
		{"TestFoo/", "TestFoo", "x", true, false},
		{"TestFoo/bar/baz", "TestFoo", "bar", true, true},
		// Subtest with a '/' in its name still allows for copy and pasted names
		// to match.
		{"TestFoo/bar/baz", "TestFoo", "bar/baz", true, false},
		{"TestFoo/bar/baz", "TestFoo/bar", "baz", true, false},
		{"TestFoo/bar/baz", "TestFoo", "x", false, false},
		{"TestFoo", "TestBar", "x", false, false},
		{"TestFoo/", "TestBar", "x", false, false},
		{"TestFoo/bar/baz", "TestBar", "x/bar/baz", false, false},

		// subtests only
		{"", "TestFoo", "x", true, false},
		{"/", "TestFoo", "x", true, false},
		{"./", "TestFoo", "x", true, false},
		{"./.", "TestFoo", "x", true, false},
		{"/bar/baz", "TestFoo", "bar", true, true},
		{"/bar/baz", "TestFoo", "bar/baz", true, false},
		{"//baz", "TestFoo", "bar/baz", true, false},
		{"//", "TestFoo", "bar/baz", true, false},
		{"/bar/baz", "TestFoo/bar", "baz", true, false},
		{"//foo", "TestFoo", "bar/baz", false, false},
		{"/bar/baz", "TestFoo", "x", false, false},
		{"/bar/baz", "TestBar", "x/bar/baz", false, false},
	}

	for _, tc := range testCases {
		m := newMatcher(regexp.MatchString, tc.pattern, "-test.run")

		parent := &common{name: tc.parent}
		if tc.parent != "" {
			parent.level = 1
		}
		if n, ok, partial := m.fullName(parent, tc.sub); ok != tc.ok || partial != tc.partial {
			t.Errorf("for pattern %q, fullName(parent=%q, sub=%q) = %q, ok %v partial %v; want ok %v partial %v",
				tc.pattern, tc.parent, tc.sub, n, ok, partial, tc.ok, tc.partial)
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
		if got, _, _ := m.fullName(parent, tc.name); got != tc.want {
			t.Errorf("%d:%s: got %q; want %q", i, tc.name, got, tc.want)
		}
	}
}
