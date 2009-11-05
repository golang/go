// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strings_test

import (
	. "strings";
	"testing";
	"unicode";
	"utf8";
)

func eq(a, b []string) bool {
	if len(a) != len(b) {
		return false;
	}
	for i := 0; i < len(a); i++ {
		if a[i] != b[i] {
			return false;
		}
	}
	return true;
}

var abcd = "abcd"
var faces = "☺☻☹"
var commas = "1,2,3,4"
var dots = "1....2....3....4"

type IndexTest struct {
	s	string;
	sep	string;
	out	int;
}

var indexTests = []IndexTest{
	IndexTest{"", "", 0},
	IndexTest{"", "a", -1},
	IndexTest{"", "foo", -1},
	IndexTest{"fo", "foo", -1},
	IndexTest{"foo", "foo", 0},
	IndexTest{"oofofoofooo", "f", 2},
	IndexTest{"oofofoofooo", "foo", 4},
	IndexTest{"barfoobarfoo", "foo", 3},
	IndexTest{"foo", "", 0},
	IndexTest{"foo", "o", 1},
	IndexTest{"abcABCabc", "A", 3},
}

var lastIndexTests = []IndexTest{
	IndexTest{"", "", 0},
	IndexTest{"", "a", -1},
	IndexTest{"", "foo", -1},
	IndexTest{"fo", "foo", -1},
	IndexTest{"foo", "foo", 0},
	IndexTest{"oofofoofooo", "f", 7},
	IndexTest{"oofofoofooo", "foo", 7},
	IndexTest{"barfoobarfoo", "foo", 9},
	IndexTest{"foo", "", 3},
	IndexTest{"foo", "o", 2},
	IndexTest{"abcABCabc", "A", 3},
	IndexTest{"abcABCabc", "a", 6},
}

// Execute f on each test case.  funcName should be the name of f; it's used
// in failure reports.
func runIndexTests(t *testing.T, f func(s, sep string) int, funcName string, testCases []IndexTest) {
	for _, test := range testCases {
		actual := f(test.s, test.sep);
		if actual != test.out {
			t.Errorf("%s(%q,%q) = %v; want %v", funcName, test.s, test.sep, actual, test.out);
		}
	}
}

func TestIndex(t *testing.T) {
	runIndexTests(t, Index, "Index", indexTests);
}

func TestLastIndex(t *testing.T) {
	runIndexTests(t, LastIndex, "LastIndex", lastIndexTests);
}


type ExplodeTest struct {
	s	string;
	n	int;
	a	[]string;
}

var explodetests = []ExplodeTest{
	ExplodeTest{abcd, 4, []string{"a", "b", "c", "d"}},
	ExplodeTest{faces, 3, []string{"☺", "☻", "☹"}},
	ExplodeTest{abcd, 2, []string{"a", "bcd"}},
}

func TestExplode(t *testing.T) {
	for _, tt := range explodetests {
		a := Split(tt.s, "", tt.n);
		if !eq(a, tt.a) {
			t.Errorf("explode(%q, %d) = %v; want %v", tt.s, tt.n, a, tt.a);
			continue;
		}
		s := Join(a, "");
		if s != tt.s {
			t.Errorf(`Join(explode(%q, %d), "") = %q`, tt.s, tt.n, s);
		}
	}
}

type SplitTest struct {
	s	string;
	sep	string;
	n	int;
	a	[]string;
}

var splittests = []SplitTest{
	SplitTest{abcd, "a", 0, []string{"", "bcd"}},
	SplitTest{abcd, "z", 0, []string{"abcd"}},
	SplitTest{abcd, "", 0, []string{"a", "b", "c", "d"}},
	SplitTest{commas, ",", 0, []string{"1", "2", "3", "4"}},
	SplitTest{dots, "...", 0, []string{"1", ".2", ".3", ".4"}},
	SplitTest{faces, "☹", 0, []string{"☺☻", ""}},
	SplitTest{faces, "~", 0, []string{faces}},
	SplitTest{faces, "", 0, []string{"☺", "☻", "☹"}},
	SplitTest{"1 2 3 4", " ", 3, []string{"1", "2", "3 4"}},
	SplitTest{"1 2", " ", 3, []string{"1", "2"}},
	SplitTest{"123", "", 2, []string{"1", "23"}},
	SplitTest{"123", "", 17, []string{"1", "2", "3"}},
}

func TestSplit(t *testing.T) {
	for _, tt := range splittests {
		a := Split(tt.s, tt.sep, tt.n);
		if !eq(a, tt.a) {
			t.Errorf("Split(%q, %q, %d) = %v; want %v", tt.s, tt.sep, tt.n, a, tt.a);
			continue;
		}
		s := Join(a, tt.sep);
		if s != tt.s {
			t.Errorf("Join(Split(%q, %q, %d), %q) = %q", tt.s, tt.sep, tt.n, tt.sep, s);
		}
	}
}

var splitaftertests = []SplitTest{
	SplitTest{abcd, "a", 0, []string{"a", "bcd"}},
	SplitTest{abcd, "z", 0, []string{"abcd"}},
	SplitTest{abcd, "", 0, []string{"a", "b", "c", "d"}},
	SplitTest{commas, ",", 0, []string{"1,", "2,", "3,", "4"}},
	SplitTest{dots, "...", 0, []string{"1...", ".2...", ".3...", ".4"}},
	SplitTest{faces, "☹", 0, []string{"☺☻☹", ""}},
	SplitTest{faces, "~", 0, []string{faces}},
	SplitTest{faces, "", 0, []string{"☺", "☻", "☹"}},
	SplitTest{"1 2 3 4", " ", 3, []string{"1 ", "2 ", "3 4"}},
	SplitTest{"1 2 3", " ", 3, []string{"1 ", "2 ", "3"}},
	SplitTest{"1 2", " ", 3, []string{"1 ", "2"}},
	SplitTest{"123", "", 2, []string{"1", "23"}},
	SplitTest{"123", "", 17, []string{"1", "2", "3"}},
}

func TestSplitAfter(t *testing.T) {
	for _, tt := range splitaftertests {
		a := SplitAfter(tt.s, tt.sep, tt.n);
		if !eq(a, tt.a) {
			t.Errorf(`Split(%q, %q, %d) = %v; want %v`, tt.s, tt.sep, tt.n, a, tt.a);
			continue;
		}
		s := Join(a, "");
		if s != tt.s {
			t.Errorf(`Join(Split(%q, %q, %d), %q) = %q`, tt.s, tt.sep, tt.n, tt.sep, s);
		}
	}
}

// Test case for any function which accepts and returns a single string.
type StringTest struct {
	in, out string;
}

// Execute f on each test case.  funcName should be the name of f; it's used
// in failure reports.
func runStringTests(t *testing.T, f func(string) string, funcName string, testCases []StringTest) {
	for _, tc := range testCases {
		actual := f(tc.in);
		if actual != tc.out {
			t.Errorf("%s(%q) = %q; want %q", funcName, tc.in, actual, tc.out);
		}
	}
}

var upperTests = []StringTest{
	StringTest{"", ""},
	StringTest{"abc", "ABC"},
	StringTest{"AbC123", "ABC123"},
	StringTest{"azAZ09_", "AZAZ09_"},
	StringTest{"\u0250\u0250\u0250\u0250\u0250", "\u2C6F\u2C6F\u2C6F\u2C6F\u2C6F"},	// grows one byte per char
}

var lowerTests = []StringTest{
	StringTest{"", ""},
	StringTest{"abc", "abc"},
	StringTest{"AbC123", "abc123"},
	StringTest{"azAZ09_", "azaz09_"},
	StringTest{"\u2C6D\u2C6D\u2C6D\u2C6D\u2C6D", "\u0251\u0251\u0251\u0251\u0251"},	// shrinks one byte per char
}

const space = "\t\v\r\f\n\u0085\u00a0\u2000\u3000"

var trimSpaceTests = []StringTest{
	StringTest{"", ""},
	StringTest{"abc", "abc"},
	StringTest{space+"abc"+space, "abc"},
	StringTest{" ", ""},
	StringTest{" \t\r\n \t\t\r\r\n\n ", ""},
	StringTest{" \t\r\n x\t\t\r\r\n\n ", "x"},
	StringTest{" \u2000\t\r\n x\t\t\r\r\ny\n \u3000", "x\t\t\r\r\ny"},
	StringTest{"1 \t\r\n2", "1 \t\r\n2"},
	StringTest{" x\x80", "x\x80"},	// invalid UTF-8 on end
	StringTest{" x\xc0", "x\xc0"},	// invalid UTF-8 on end
}

func tenRunes(rune int) string {
	r := make([]int, 10);
	for i := range r {
		r[i] = rune;
	}
	return string(r);
}

func TestMap(t *testing.T) {
	// Run a couple of awful growth/shrinkage tests
	a := tenRunes('a');
	// 1.  Grow.  This triggers two reallocations in Map.
	maxRune := func(rune int) int { return unicode.MaxRune };
	m := Map(maxRune, a);
	expect := tenRunes(unicode.MaxRune);
	if m != expect {
		t.Errorf("growing: expected %q got %q", expect, m);
	}
	// 2. Shrink
	minRune := func(rune int) int { return 'a' };
	m = Map(minRune, tenRunes(unicode.MaxRune));
	expect = a;
	if m != expect {
		t.Errorf("shrinking: expected %q got %q", expect, m);
	}
}

func TestToUpper(t *testing.T) {
	runStringTests(t, ToUpper, "ToUpper", upperTests);
}

func TestToLower(t *testing.T) {
	runStringTests(t, ToLower, "ToLower", lowerTests);
}

func TestTrimSpace(t *testing.T) {
	runStringTests(t, TrimSpace, "TrimSpace", trimSpaceTests);
}

func equal(m string, s1, s2 string, t *testing.T) bool {
	if s1 == s2 {
		return true;
	}
	e1 := Split(s1, "", 0);
	e2 := Split(s2, "", 0);
	for i, c1 := range e1 {
		if i > len(e2) {
			break;
		}
		r1, _ := utf8.DecodeRuneInString(c1);
		r2, _ := utf8.DecodeRuneInString(e2[i]);
		if r1 != r2 {
			t.Errorf("%s diff at %d: U+%04X U+%04X", m, i, r1, r2);
		}
	}
	return false;
}

func TestCaseConsistency(t *testing.T) {
	// Make a string of all the runes.
	a := make([]int, unicode.MaxRune + 1);
	for i := range a {
		a[i] = i;
	}
	s := string(a);
	// convert the cases.
	upper := ToUpper(s);
	lower := ToLower(s);

	// Consistency checks
	if n := utf8.RuneCountInString(upper); n != unicode.MaxRune + 1 {
		t.Error("rune count wrong in upper:", n);
	}
	if n := utf8.RuneCountInString(lower); n != unicode.MaxRune + 1 {
		t.Error("rune count wrong in lower:", n);
	}
	if !equal("ToUpper(upper)", ToUpper(upper), upper, t) {
		t.Error("ToUpper(upper) consistency fail");
	}
	if !equal("ToLower(lower)", ToLower(lower), lower, t) {
		t.Error("ToLower(lower) consistency fail");
	}
	/*
		  These fail because of non-one-to-oneness of the data, such as multiple
		  upper case 'I' mapping to 'i'.  We comment them out but keep them for
		  interest.
		  For instance: CAPITAL LETTER I WITH DOT ABOVE:
			unicode.ToUpper(unicode.ToLower('\u0130')) != '\u0130'

		if !equal("ToUpper(lower)", ToUpper(lower), upper, t) {
			t.Error("ToUpper(lower) consistency fail");
		}
		if !equal("ToLower(upper)", ToLower(upper), lower, t) {
			t.Error("ToLower(upper) consistency fail");
		}
	*/
}
