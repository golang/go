// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testing

import (
	"fmt"
	"reflect"
	"regexp"
	"strings"
	"unicode"
)

func init() {
	testingTesting = true
}

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
	res := func(s ...string) filterMatch { return simpleMatch(s) }
	alt := func(m ...filterMatch) filterMatch { return alternationMatch(m) }
	testCases := []struct {
		pattern string
		result  filterMatch
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

		{"A/B|C/D", alt(res("A", "B"), res("C", "D"))},

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
			if err := a.verify("", regexp.MatchString); err != nil {
				ok = false
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
		skip        string
		parent, sub string
		ok          bool
		partial     bool
	}{
		// Behavior without subtests.
		{"", "", "", "TestFoo", true, false},
		{"TestFoo", "", "", "TestFoo", true, false},
		{"TestFoo/", "", "", "TestFoo", true, true},
		{"TestFoo/bar/baz", "", "", "TestFoo", true, true},
		{"TestFoo", "", "", "TestBar", false, false},
		{"TestFoo/", "", "", "TestBar", false, false},
		{"TestFoo/bar/baz", "", "", "TestBar/bar/baz", false, false},
		{"", "TestBar", "", "TestFoo", true, false},
		{"", "TestBar", "", "TestBar", false, false},

		// Skipping a non-existent test doesn't change anything.
		{"", "TestFoo/skipped", "", "TestFoo", true, false},
		{"TestFoo", "TestFoo/skipped", "", "TestFoo", true, false},
		{"TestFoo/", "TestFoo/skipped", "", "TestFoo", true, true},
		{"TestFoo/bar/baz", "TestFoo/skipped", "", "TestFoo", true, true},
		{"TestFoo", "TestFoo/skipped", "", "TestBar", false, false},
		{"TestFoo/", "TestFoo/skipped", "", "TestBar", false, false},
		{"TestFoo/bar/baz", "TestFoo/skipped", "", "TestBar/bar/baz", false, false},

		// with subtests
		{"", "", "TestFoo", "x", true, false},
		{"TestFoo", "", "TestFoo", "x", true, false},
		{"TestFoo/", "", "TestFoo", "x", true, false},
		{"TestFoo/bar/baz", "", "TestFoo", "bar", true, true},

		{"", "TestFoo/skipped", "TestFoo", "x", true, false},
		{"TestFoo", "TestFoo/skipped", "TestFoo", "x", true, false},
		{"TestFoo", "TestFoo/skipped", "TestFoo", "skipped", false, false},
		{"TestFoo/", "TestFoo/skipped", "TestFoo", "x", true, false},
		{"TestFoo/bar/baz", "TestFoo/skipped", "TestFoo", "bar", true, true},

		// Subtest with a '/' in its name still allows for copy and pasted names
		// to match.
		{"TestFoo/bar/baz", "", "TestFoo", "bar/baz", true, false},
		{"TestFoo/bar/baz", "TestFoo/bar/baz", "TestFoo", "bar/baz", false, false},
		{"TestFoo/bar/baz", "TestFoo/bar/baz/skip", "TestFoo", "bar/baz", true, false},
		{"TestFoo/bar/baz", "", "TestFoo/bar", "baz", true, false},
		{"TestFoo/bar/baz", "", "TestFoo", "x", false, false},
		{"TestFoo", "", "TestBar", "x", false, false},
		{"TestFoo/", "", "TestBar", "x", false, false},
		{"TestFoo/bar/baz", "", "TestBar", "x/bar/baz", false, false},

		{"A/B|C/D", "", "TestA", "B", true, false},
		{"A/B|C/D", "", "TestC", "D", true, false},
		{"A/B|C/D", "", "TestA", "C", false, false},

		// subtests only
		{"", "", "TestFoo", "x", true, false},
		{"/", "", "TestFoo", "x", true, false},
		{"./", "", "TestFoo", "x", true, false},
		{"./.", "", "TestFoo", "x", true, false},
		{"/bar/baz", "", "TestFoo", "bar", true, true},
		{"/bar/baz", "", "TestFoo", "bar/baz", true, false},
		{"//baz", "", "TestFoo", "bar/baz", true, false},
		{"//", "", "TestFoo", "bar/baz", true, false},
		{"/bar/baz", "", "TestFoo/bar", "baz", true, false},
		{"//foo", "", "TestFoo", "bar/baz", false, false},
		{"/bar/baz", "", "TestFoo", "x", false, false},
		{"/bar/baz", "", "TestBar", "x/bar/baz", false, false},
	}

	for _, tc := range testCases {
		m := newMatcher(regexp.MatchString, tc.pattern, "-test.run", tc.skip)

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

var namingTestCases = []struct{ name, want string }{
	// Uniqueness
	{"", "x/#00"},
	{"", "x/#01"},
	{"#0", "x/#0"},      // Doesn't conflict with #00 because the number of digits differs.
	{"#00", "x/#00#01"}, // Conflicts with implicit #00 (used above), so add a suffix.
	{"#", "x/#"},
	{"#", "x/##01"},

	{"t", "x/t"},
	{"t", "x/t#01"},
	{"t", "x/t#02"},
	{"t#00", "x/t#00"}, // Explicit "#00" doesn't conflict with the unsuffixed first subtest.

	{"a#01", "x/a#01"},    // user has subtest with this name.
	{"a", "x/a"},          // doesn't conflict with this name.
	{"a", "x/a#02"},       // This string is claimed now, so resume
	{"a", "x/a#03"},       // with counting.
	{"a#02", "x/a#02#01"}, // We already used a#02 once, so add a suffix.

	{"b#00", "x/b#00"},
	{"b", "x/b"}, // Implicit 0 doesn't conflict with explicit "#00".
	{"b", "x/b#01"},
	{"b#9223372036854775807", "x/b#9223372036854775807"}, // MaxInt64
	{"b", "x/b#02"},
	{"b", "x/b#03"},

	// Sanitizing
	{"A:1 B:2", "x/A:1_B:2"},
	{"s\t\r\u00a0", "x/s___"},
	{"\x01", `x/\x01`},
	{"\U0010ffff", `x/\U0010ffff`},
}

func TestNaming(t *T) {
	m := newMatcher(regexp.MatchString, "", "", "")
	parent := &common{name: "x", level: 1} // top-level test.

	for i, tc := range namingTestCases {
		if got, _, _ := m.fullName(parent, tc.name); got != tc.want {
			t.Errorf("%d:%s: got %q; want %q", i, tc.name, got, tc.want)
		}
	}
}

func FuzzNaming(f *F) {
	for _, tc := range namingTestCases {
		f.Add(tc.name)
	}
	parent := &common{name: "x", level: 1}
	var m *matcher
	var seen map[string]string
	reset := func() {
		m = allMatcher()
		seen = make(map[string]string)
	}
	reset()

	f.Fuzz(func(t *T, subname string) {
		if len(subname) > 10 {
			// Long names attract the OOM killer.
			t.Skip()
		}
		name := m.unique(parent.name, subname)
		if !strings.Contains(name, "/"+subname) {
			t.Errorf("name %q does not contain subname %q", name, subname)
		}
		if prev, ok := seen[name]; ok {
			t.Errorf("name %q generated by both %q and %q", name, prev, subname)
		}
		if len(seen) > 1e6 {
			// Free up memory.
			reset()
		}
		seen[name] = subname
	})
}

// GoString returns a string that is more readable than the default, which makes
// it easier to read test errors.
func (m alternationMatch) GoString() string {
	s := make([]string, len(m))
	for i, m := range m {
		s[i] = fmt.Sprintf("%#v", m)
	}
	return fmt.Sprintf("(%s)", strings.Join(s, " | "))
}
