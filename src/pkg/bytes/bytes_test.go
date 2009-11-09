// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bytes_test

import (
	. "bytes";
	"strings";
	"testing";
	"unicode";
)

func eq(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := 0; i < len(a); i++ {
		if a[i] != b[i] {
			return false
		}
	}
	return true;
}

func arrayOfString(a [][]byte) []string {
	result := make([]string, len(a));
	for j := 0; j < len(a); j++ {
		result[j] = string(a[j])
	}
	return result;
}

// For ease of reading, the test cases use strings that are converted to byte
// arrays before invoking the functions.

var abcd = "abcd"
var faces = "☺☻☹"
var commas = "1,2,3,4"
var dots = "1....2....3....4"

type CompareTest struct {
	a	string;
	b	string;
	cmp	int;
}

var comparetests = []CompareTest{
	CompareTest{"", "", 0},
	CompareTest{"a", "", 1},
	CompareTest{"", "a", -1},
	CompareTest{"abc", "abc", 0},
	CompareTest{"ab", "abc", -1},
	CompareTest{"abc", "ab", 1},
	CompareTest{"x", "ab", 1},
	CompareTest{"ab", "x", -1},
	CompareTest{"x", "a", 1},
	CompareTest{"b", "x", -1},
}

func TestCompare(t *testing.T) {
	for i := 0; i < len(comparetests); i++ {
		tt := comparetests[i];
		a := strings.Bytes(tt.a);
		b := strings.Bytes(tt.b);
		cmp := Compare(a, b);
		eql := Equal(a, b);
		if cmp != tt.cmp {
			t.Errorf(`Compare(%q, %q) = %v`, tt.a, tt.b, cmp)
		}
		if eql != (tt.cmp == 0) {
			t.Errorf(`Equal(%q, %q) = %v`, tt.a, tt.b, eql)
		}
	}
}


type ExplodeTest struct {
	s	string;
	n	int;
	a	[]string;
}

var explodetests = []ExplodeTest{
	ExplodeTest{abcd, 0, []string{"a", "b", "c", "d"}},
	ExplodeTest{faces, 0, []string{"☺", "☻", "☹"}},
	ExplodeTest{abcd, 2, []string{"a", "bcd"}},
}

func TestExplode(t *testing.T) {
	for _, tt := range (explodetests) {
		a := Split(strings.Bytes(tt.s), nil, tt.n);
		result := arrayOfString(a);
		if !eq(result, tt.a) {
			t.Errorf(`Explode("%s", %d) = %v; want %v`, tt.s, tt.n, result, tt.a);
			continue;
		}
		s := Join(a, []byte{});
		if string(s) != tt.s {
			t.Errorf(`Join(Explode("%s", %d), "") = "%s"`, tt.s, tt.n, s)
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
	SplitTest{"1 2 3", " ", 3, []string{"1", "2", "3"}},
	SplitTest{"1 2", " ", 3, []string{"1", "2"}},
	SplitTest{"123", "", 2, []string{"1", "23"}},
	SplitTest{"123", "", 17, []string{"1", "2", "3"}},
}

func TestSplit(t *testing.T) {
	for _, tt := range splittests {
		a := Split(strings.Bytes(tt.s), strings.Bytes(tt.sep), tt.n);
		result := arrayOfString(a);
		if !eq(result, tt.a) {
			t.Errorf(`Split(%q, %q, %d) = %v; want %v`, tt.s, tt.sep, tt.n, result, tt.a);
			continue;
		}
		s := Join(a, strings.Bytes(tt.sep));
		if string(s) != tt.s {
			t.Errorf(`Join(Split(%q, %q, %d), %q) = %q`, tt.s, tt.sep, tt.n, tt.sep, s)
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
		a := SplitAfter(strings.Bytes(tt.s), strings.Bytes(tt.sep), tt.n);
		result := arrayOfString(a);
		if !eq(result, tt.a) {
			t.Errorf(`Split(%q, %q, %d) = %v; want %v`, tt.s, tt.sep, tt.n, result, tt.a);
			continue;
		}
		s := Join(a, nil);
		if string(s) != tt.s {
			t.Errorf(`Join(Split(%q, %q, %d), %q) = %q`, tt.s, tt.sep, tt.n, tt.sep, s)
		}
	}
}

type CopyTest struct {
	a	string;
	b	string;
	n	int;
	res	string;
}

var copytests = []CopyTest{
	CopyTest{"", "", 0, ""},
	CopyTest{"a", "", 0, "a"},
	CopyTest{"a", "a", 1, "a"},
	CopyTest{"a", "b", 1, "b"},
	CopyTest{"xyz", "abc", 3, "abc"},
	CopyTest{"wxyz", "abc", 3, "abcz"},
	CopyTest{"xyz", "abcd", 3, "abc"},
}

func TestCopy(t *testing.T) {
	for i := 0; i < len(copytests); i++ {
		tt := copytests[i];
		dst := strings.Bytes(tt.a);
		n := Copy(dst, strings.Bytes(tt.b));
		result := string(dst);
		if result != tt.res || n != tt.n {
			t.Errorf(`Copy(%q, %q) = %d, %q; want %d, %q`, tt.a, tt.b, n, result, tt.n, tt.res);
			continue;
		}
	}
}

// Test case for any function which accepts and returns a byte array.
// For ease of creation, we write the byte arrays as strings.
type StringTest struct {
	in, out string;
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

// Bytes returns a new slice containing the bytes in s.
// Borrowed from strings to avoid dependency.
func Bytes(s string) []byte {
	b := make([]byte, len(s));
	for i := 0; i < len(s); i++ {
		b[i] = s[i]
	}
	return b;
}

// Execute f on each test case.  funcName should be the name of f; it's used
// in failure reports.
func runStringTests(t *testing.T, f func([]byte) []byte, funcName string, testCases []StringTest) {
	for _, tc := range testCases {
		actual := string(f(Bytes(tc.in)));
		if actual != tc.out {
			t.Errorf("%s(%q) = %q; want %q", funcName, tc.in, actual, tc.out)
		}
	}
}

func tenRunes(rune int) string {
	r := make([]int, 10);
	for i := range r {
		r[i] = rune
	}
	return string(r);
}

func TestMap(t *testing.T) {
	// Run a couple of awful growth/shrinkage tests
	a := tenRunes('a');
	// 1.  Grow.  This triggers two reallocations in Map.
	maxRune := func(rune int) int { return unicode.MaxRune };
	m := Map(maxRune, Bytes(a));
	expect := tenRunes(unicode.MaxRune);
	if string(m) != expect {
		t.Errorf("growing: expected %q got %q", expect, m)
	}
	// 2. Shrink
	minRune := func(rune int) int { return 'a' };
	m = Map(minRune, Bytes(tenRunes(unicode.MaxRune)));
	expect = a;
	if string(m) != expect {
		t.Errorf("shrinking: expected %q got %q", expect, m)
	}
}

func TestToUpper(t *testing.T)	{ runStringTests(t, ToUpper, "ToUpper", upperTests) }

func TestToLower(t *testing.T)	{ runStringTests(t, ToLower, "ToLower", lowerTests) }

func TestTrimSpace(t *testing.T)	{ runStringTests(t, TrimSpace, "TrimSpace", trimSpaceTests) }

type AddTest struct {
	s, t	string;
	cap	int;
}

var addtests = []AddTest{
	AddTest{"", "", 0},
	AddTest{"a", "", 1},
	AddTest{"a", "b", 1},
	AddTest{"abc", "def", 100},
}

func TestAdd(t *testing.T) {
	for _, test := range addtests {
		b := make([]byte, len(test.s), test.cap);
		for i := 0; i < len(test.s); i++ {
			b[i] = test.s[i]
		}
		b = Add(b, strings.Bytes(test.t));
		if string(b) != test.s + test.t {
			t.Errorf("Add(%q,%q) = %q", test.s, test.t, string(b))
		}
	}
}

func TestAddByte(t *testing.T) {
	const N = 2e5;
	b := make([]byte, 0);
	for i := 0; i < N; i++ {
		b = AddByte(b, byte(i))
	}
	if len(b) != N {
		t.Errorf("AddByte: too small; expected %d got %d", N, len(b))
	}
	for i, c := range b {
		if c != byte(i) {
			t.Fatalf("AddByte: b[%d] should be %d is %d", i, c, byte(i))
		}
	}
}
