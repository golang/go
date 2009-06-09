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

type ExplodeTest struct {
	s string;
	a []string;
}
var explodetests = []ExplodeTest {
	ExplodeTest{ abcd,	[]string{"a", "b", "c", "d"} },
	ExplodeTest{ faces,	[]string{"☺", "☻", "☹" } },
}
func TestExplode(t *testing.T) {
	for i := 0; i < len(explodetests); i++ {
		tt := explodetests[i];
		a := Explode(tt.s);
		if !eq(a, tt.a) {
			t.Errorf("Explode(%q) = %v; want %v", tt.s, a, tt.a);
			continue;
		}
		s := Join(a, "");
		if s != tt.s {
			t.Errorf(`Join(Explode(%q), "") = %q`, tt.s, s);
		}
	}
}

type SplitTest struct {
	s string;
	sep string;
	a []string;
}
var splittests = []SplitTest {
	SplitTest{ abcd,	"a",	[]string{"", "bcd"} },
	SplitTest{ abcd,	"z",	[]string{"abcd"} },
	SplitTest{ abcd,	"",	[]string{"a", "b", "c", "d"} },
	SplitTest{ commas,	",",	[]string{"1", "2", "3", "4"} },
	SplitTest{ dots,	"...",	[]string{"1", ".2", ".3", ".4"} },
	SplitTest{ faces,	"☹",	[]string{"☺☻", ""} },
	SplitTest{ faces,	"~",	[]string{faces} },
	SplitTest{ faces,	"",	[]string{"☺", "☻", "☹"} },
}
func TestSplit(t *testing.T) {
	for i := 0; i < len(splittests); i++ {
		tt := splittests[i];
		a := Split(tt.s, tt.sep);
		if !eq(a, tt.a) {
			t.Errorf("Split(%q, %q) = %v; want %v", tt.s, tt.sep, a, tt.a);
			continue;
		}
		s := Join(a, tt.sep);
		if s != tt.s {
			t.Errorf("Join(Split(%q, %q), %q) = %q", tt.s, tt.sep, tt.sep, s);
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
		if (actual != tc.out) {
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

