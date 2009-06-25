// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strings

import (
	"strings";
	"testing";
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

var abcd = "abcd";
var faces = "☺☻☹";
var commas = "1,2,3,4";
var dots = "1....2....3....4";

type IndexTest struct {
	s string;
	sep string;
	out int;
}

var indexTests = []IndexTest {
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

var lastIndexTests = []IndexTest {
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
	for i,test := range testCases {
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
	s string;
	n int;
	a []string;
}
var explodetests = []ExplodeTest {
	ExplodeTest{ abcd,	4, []string{"a", "b", "c", "d"} },
	ExplodeTest{ faces,	3, []string{"☺", "☻", "☹"} },
	ExplodeTest{ abcd,	2, []string{"a", "bcd"} },
}
func TestExplode(t *testing.T) {
	for _, tt := range explodetests {
		a := explode(tt.s, tt.n);
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
	s string;
	sep string;
	n int;
	a []string;
}
var splittests = []SplitTest {
	SplitTest{ abcd,	"a",	0, []string{"", "bcd"} },
	SplitTest{ abcd,	"z",	0, []string{"abcd"} },
	SplitTest{ abcd,	"",	0, []string{"a", "b", "c", "d"} },
	SplitTest{ commas,	",",	0, []string{"1", "2", "3", "4"} },
	SplitTest{ dots,	"...",	0, []string{"1", ".2", ".3", ".4"} },
	SplitTest{ faces,	"☹",	0, []string{"☺☻", ""} },
	SplitTest{ faces,	"~",	0, []string{faces} },
	SplitTest{ faces,	"",	0, []string{"☺", "☻", "☹"} },
	SplitTest{ "1 2 3 4",	" ",	3, []string{"1", "2", "3 4"} },
	SplitTest{ "1 2",	" ",	3, []string{"1", "2"} },
	SplitTest{ "123",	"",	2, []string{"1", "23"} },
	SplitTest{ "123",	"",	17, []string{"1", "2", "3"} },
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

// Test case for any function which accepts and returns a single string.
type StringTest struct {
	in, out string;
}

// Execute f on each test case.  funcName should be the name of f; it's used
// in failure reports.
func runStringTests(t *testing.T, f func(string) string, funcName string, testCases []StringTest) {
	for i, tc := range testCases {
		actual := f(tc.in);
		if actual != tc.out {
			t.Errorf("%s(%q) = %q; want %q", funcName, tc.in, actual, tc.out);
		}
	}
}

var upperASCIITests = []StringTest {
	StringTest{"", ""},
	StringTest{"abc", "ABC"},
	StringTest{"AbC123", "ABC123"},
	StringTest{"azAZ09_", "AZAZ09_"}
}

var lowerASCIITests = []StringTest {
	StringTest{"", ""},
	StringTest{"abc", "abc"},
	StringTest{"AbC123", "abc123"},
	StringTest{"azAZ09_", "azaz09_"}
}

var trimSpaceASCIITests = []StringTest {
	StringTest{"", ""},
	StringTest{"abc", "abc"},
	StringTest{" ", ""},
	StringTest{" \t\r\n \t\t\r\r\n\n ", ""},
	StringTest{" \t\r\n x\t\t\r\r\n\n ", "x"},
	StringTest{" \t\r\n x\t\t\r\r\ny\n ", "x\t\t\r\r\ny"},
	StringTest{"1 \t\r\n2", "1 \t\r\n2"},
}

func TestUpperASCII(t *testing.T) {
	runStringTests(t, UpperASCII, "UpperASCII", upperASCIITests);
}

func TestLowerASCII(t *testing.T) {
	runStringTests(t, LowerASCII, "LowerASCII", lowerASCIITests);
}

func TestTrimSpaceASCII(t *testing.T) {
	runStringTests(t, TrimSpaceASCII, "TrimSpaceASCII", trimSpaceASCIITests);
}

