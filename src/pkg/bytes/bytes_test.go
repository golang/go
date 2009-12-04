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

type BinOpTest struct {
	a	string;
	b	string;
	i	int;
}

var comparetests = []BinOpTest{
	BinOpTest{"", "", 0},
	BinOpTest{"a", "", 1},
	BinOpTest{"", "a", -1},
	BinOpTest{"abc", "abc", 0},
	BinOpTest{"ab", "abc", -1},
	BinOpTest{"abc", "ab", 1},
	BinOpTest{"x", "ab", 1},
	BinOpTest{"ab", "x", -1},
	BinOpTest{"x", "a", 1},
	BinOpTest{"b", "x", -1},
}

func TestCompare(t *testing.T) {
	for _, tt := range comparetests {
		a := strings.Bytes(tt.a);
		b := strings.Bytes(tt.b);
		cmp := Compare(a, b);
		eql := Equal(a, b);
		if cmp != tt.i {
			t.Errorf(`Compare(%q, %q) = %v`, tt.a, tt.b, cmp)
		}
		if eql != (tt.i == 0) {
			t.Errorf(`Equal(%q, %q) = %v`, tt.a, tt.b, eql)
		}
	}
}

var indextests = []BinOpTest{
	BinOpTest{"", "", 0},
	BinOpTest{"a", "", 0},
	BinOpTest{"", "a", -1},
	BinOpTest{"abc", "abc", 0},
	BinOpTest{"ab", "abc", -1},
	BinOpTest{"abc", "bc", 1},
	BinOpTest{"x", "ab", -1},
	// one-byte tests for IndexByte
	BinOpTest{"ab", "x", -1},
	BinOpTest{"", "a", -1},
	BinOpTest{"x", "a", -1},
	BinOpTest{"x", "x", 0},
	BinOpTest{"abc", "a", 0},
	BinOpTest{"abc", "b", 1},
	BinOpTest{"abc", "c", 2},
	BinOpTest{"abc", "x", -1},
}

func TestIndex(t *testing.T) {
	for _, tt := range indextests {
		a := strings.Bytes(tt.a);
		b := strings.Bytes(tt.b);
		pos := Index(a, b);
		if pos != tt.i {
			t.Errorf(`Index(%q, %q) = %v`, tt.a, tt.b, pos)
		}
	}
}

func TestIndexByte(t *testing.T) {
	for _, tt := range indextests {
		if len(tt.b) != 1 {
			continue
		}
		a := strings.Bytes(tt.a);
		b := tt.b[0];
		pos := IndexByte(a, b);
		if pos != tt.i {
			t.Errorf(`IndexByte(%q, '%c') = %v`, tt.a, b, pos)
		}
		posp := IndexBytePortable(a, b);
		if posp != tt.i {
			t.Errorf(`indexBytePortable(%q, '%c') = %v`, tt.a, b, posp)
		}
	}
}

func BenchmarkIndexByte4K(b *testing.B)	{ bmIndex(b, IndexByte, 4<<10) }

func BenchmarkIndexByte4M(b *testing.B)	{ bmIndex(b, IndexByte, 4<<20) }

func BenchmarkIndexByte64M(b *testing.B)	{ bmIndex(b, IndexByte, 64<<20) }

func BenchmarkIndexBytePortable4K(b *testing.B) {
	bmIndex(b, IndexBytePortable, 4<<10)
}

func BenchmarkIndexBytePortable4M(b *testing.B) {
	bmIndex(b, IndexBytePortable, 4<<20)
}

func BenchmarkIndexBytePortable64M(b *testing.B) {
	bmIndex(b, IndexBytePortable, 64<<20)
}

var bmbuf []byte

func bmIndex(b *testing.B, index func([]byte, byte) int, n int) {
	if len(bmbuf) < n {
		bmbuf = make([]byte, n)
	}
	b.SetBytes(int64(n));
	buf := bmbuf[0:n];
	buf[n-1] = 'x';
	for i := 0; i < b.N; i++ {
		j := index(buf, 'x');
		if j != n-1 {
			panic("bad index", j)
		}
	}
	buf[n-1] = '0';
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
	StringTest{space + "abc" + space, "abc"},
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

// User-defined self-inverse mapping function
func rot13(rune int) int {
	step := 13;
	if rune >= 'a' && rune <= 'z' {
		return ((rune - 'a' + step) % 26) + 'a'
	}
	if rune >= 'A' && rune <= 'Z' {
		return ((rune - 'A' + step) % 26) + 'A'
	}
	return rune;
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

	// 3. Rot13
	m = Map(rot13, Bytes("a to zed"));
	expect = "n gb mrq";
	if string(m) != expect {
		t.Errorf("rot13: expected %q got %q", expect, m)
	}

	// 4. Rot13^2
	m = Map(rot13, Map(rot13, Bytes("a to zed")));
	expect = "a to zed";
	if string(m) != expect {
		t.Errorf("rot13: expected %q got %q", expect, m)
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
		if string(b) != test.s+test.t {
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

type RepeatTest struct {
	in, out	string;
	count	int;
}

var RepeatTests = []RepeatTest{
	RepeatTest{"", "", 0},
	RepeatTest{"", "", 1},
	RepeatTest{"", "", 2},
	RepeatTest{"-", "", 0},
	RepeatTest{"-", "-", 1},
	RepeatTest{"-", "----------", 10},
	RepeatTest{"abc ", "abc abc abc ", 3},
}

func TestRepeat(t *testing.T) {
	for _, tt := range RepeatTests {
		tin := strings.Bytes(tt.in);
		tout := strings.Bytes(tt.out);
		a := Repeat(tin, tt.count);
		if !Equal(a, tout) {
			t.Errorf("Repeat(%q, %d) = %q; want %q", tin, tt.count, a, tout);
			continue;
		}
	}
}

func runesEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i, r := range a {
		if r != b[i] {
			return false
		}
	}
	return true;
}

type RunesTest struct {
	in	string;
	out	[]int;
	lossy	bool;
}

var RunesTests = []RunesTest{
	RunesTest{"", []int{}, false},
	RunesTest{" ", []int{32}, false},
	RunesTest{"ABC", []int{65, 66, 67}, false},
	RunesTest{"abc", []int{97, 98, 99}, false},
	RunesTest{"\u65e5\u672c\u8a9e", []int{26085, 26412, 35486}, false},
	RunesTest{"ab\x80c", []int{97, 98, 0xFFFD, 99}, true},
	RunesTest{"ab\xc0c", []int{97, 98, 0xFFFD, 99}, true},
}

func TestRunes(t *testing.T) {
	for _, tt := range RunesTests {
		tin := strings.Bytes(tt.in);
		a := Runes(tin);
		if !runesEqual(a, tt.out) {
			t.Errorf("Runes(%q) = %v; want %v", tin, a, tt.out);
			continue;
		}
		if !tt.lossy {
			// can only test reassembly if we didn't lose information
			s := string(a);
			if s != tt.in {
				t.Errorf("string(Runes(%q)) = %x; want %x", tin, s, tin)
			}
		}
	}
}
