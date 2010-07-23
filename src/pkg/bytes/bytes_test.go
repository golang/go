// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bytes_test

import (
	. "bytes"
	"testing"
	"unicode"
	"utf8"
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
	return true
}

func arrayOfString(a [][]byte) []string {
	result := make([]string, len(a))
	for j := 0; j < len(a); j++ {
		result[j] = string(a[j])
	}
	return result
}

// For ease of reading, the test cases use strings that are converted to byte
// arrays before invoking the functions.

var abcd = "abcd"
var faces = "☺☻☹"
var commas = "1,2,3,4"
var dots = "1....2....3....4"

type BinOpTest struct {
	a string
	b string
	i int
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
		a := []byte(tt.a)
		b := []byte(tt.b)
		cmp := Compare(a, b)
		eql := Equal(a, b)
		if cmp != tt.i {
			t.Errorf(`Compare(%q, %q) = %v`, tt.a, tt.b, cmp)
		}
		if eql != (tt.i == 0) {
			t.Errorf(`Equal(%q, %q) = %v`, tt.a, tt.b, eql)
		}
	}
}

var indexTests = []BinOpTest{
	BinOpTest{"", "", 0},
	BinOpTest{"", "a", -1},
	BinOpTest{"", "foo", -1},
	BinOpTest{"fo", "foo", -1},
	BinOpTest{"foo", "foo", 0},
	BinOpTest{"oofofoofooo", "f", 2},
	BinOpTest{"oofofoofooo", "foo", 4},
	BinOpTest{"barfoobarfoo", "foo", 3},
	BinOpTest{"foo", "", 0},
	BinOpTest{"foo", "o", 1},
	BinOpTest{"abcABCabc", "A", 3},
	// cases with one byte strings - test IndexByte and special case in Index()
	BinOpTest{"", "a", -1},
	BinOpTest{"x", "a", -1},
	BinOpTest{"x", "x", 0},
	BinOpTest{"abc", "a", 0},
	BinOpTest{"abc", "b", 1},
	BinOpTest{"abc", "c", 2},
	BinOpTest{"abc", "x", -1},
}

var lastIndexTests = []BinOpTest{
	BinOpTest{"", "", 0},
	BinOpTest{"", "a", -1},
	BinOpTest{"", "foo", -1},
	BinOpTest{"fo", "foo", -1},
	BinOpTest{"foo", "foo", 0},
	BinOpTest{"foo", "f", 0},
	BinOpTest{"oofofoofooo", "f", 7},
	BinOpTest{"oofofoofooo", "foo", 7},
	BinOpTest{"barfoobarfoo", "foo", 9},
	BinOpTest{"foo", "", 3},
	BinOpTest{"foo", "o", 2},
	BinOpTest{"abcABCabc", "A", 3},
	BinOpTest{"abcABCabc", "a", 6},
}

var indexAnyTests = []BinOpTest{
	BinOpTest{"", "", -1},
	BinOpTest{"", "a", -1},
	BinOpTest{"", "abc", -1},
	BinOpTest{"a", "", -1},
	BinOpTest{"a", "a", 0},
	BinOpTest{"aaa", "a", 0},
	BinOpTest{"abc", "xyz", -1},
	BinOpTest{"abc", "xcz", 2},
	BinOpTest{"ab☺c", "x☺yz", 2},
	BinOpTest{"aRegExp*", ".(|)*+?^$[]", 7},
	BinOpTest{dots + dots + dots, " ", -1},
}

// Execute f on each test case.  funcName should be the name of f; it's used
// in failure reports.
func runIndexTests(t *testing.T, f func(s, sep []byte) int, funcName string, testCases []BinOpTest) {
	for _, test := range testCases {
		a := []byte(test.a)
		b := []byte(test.b)
		actual := f(a, b)
		if actual != test.i {
			t.Errorf("%s(%q,%q) = %v; want %v", funcName, a, b, actual, test.i)
		}
	}
}

func TestIndex(t *testing.T)     { runIndexTests(t, Index, "Index", indexTests) }
func TestLastIndex(t *testing.T) { runIndexTests(t, LastIndex, "LastIndex", lastIndexTests) }
func TestIndexAny(t *testing.T) {
	for _, test := range indexAnyTests {
		a := []byte(test.a)
		actual := IndexAny(a, test.b)
		if actual != test.i {
			t.Errorf("IndexAny(%q,%q) = %v; want %v", a, test.b, actual, test.i)
		}
	}
}

func TestIndexByte(t *testing.T) {
	for _, tt := range indexTests {
		if len(tt.b) != 1 {
			continue
		}
		a := []byte(tt.a)
		b := tt.b[0]
		pos := IndexByte(a, b)
		if pos != tt.i {
			t.Errorf(`IndexByte(%q, '%c') = %v`, tt.a, b, pos)
		}
		posp := IndexBytePortable(a, b)
		if posp != tt.i {
			t.Errorf(`indexBytePortable(%q, '%c') = %v`, tt.a, b, posp)
		}
	}
}

func BenchmarkIndexByte4K(b *testing.B) { bmIndex(b, IndexByte, 4<<10) }

func BenchmarkIndexByte4M(b *testing.B) { bmIndex(b, IndexByte, 4<<20) }

func BenchmarkIndexByte64M(b *testing.B) { bmIndex(b, IndexByte, 64<<20) }

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
	b.SetBytes(int64(n))
	buf := bmbuf[0:n]
	buf[n-1] = 'x'
	for i := 0; i < b.N; i++ {
		j := index(buf, 'x')
		if j != n-1 {
			println("bad index", j)
			panic("bad index")
		}
	}
	buf[n-1] = '0'
}

type ExplodeTest struct {
	s string
	n int
	a []string
}

var explodetests = []ExplodeTest{
	ExplodeTest{abcd, -1, []string{"a", "b", "c", "d"}},
	ExplodeTest{faces, -1, []string{"☺", "☻", "☹"}},
	ExplodeTest{abcd, 2, []string{"a", "bcd"}},
}

func TestExplode(t *testing.T) {
	for _, tt := range explodetests {
		a := Split([]byte(tt.s), nil, tt.n)
		result := arrayOfString(a)
		if !eq(result, tt.a) {
			t.Errorf(`Explode("%s", %d) = %v; want %v`, tt.s, tt.n, result, tt.a)
			continue
		}
		s := Join(a, []byte{})
		if string(s) != tt.s {
			t.Errorf(`Join(Explode("%s", %d), "") = "%s"`, tt.s, tt.n, s)
		}
	}
}


type SplitTest struct {
	s   string
	sep string
	n   int
	a   []string
}

var splittests = []SplitTest{
	SplitTest{abcd, "a", 0, nil},
	SplitTest{abcd, "a", -1, []string{"", "bcd"}},
	SplitTest{abcd, "z", -1, []string{"abcd"}},
	SplitTest{abcd, "", -1, []string{"a", "b", "c", "d"}},
	SplitTest{commas, ",", -1, []string{"1", "2", "3", "4"}},
	SplitTest{dots, "...", -1, []string{"1", ".2", ".3", ".4"}},
	SplitTest{faces, "☹", -1, []string{"☺☻", ""}},
	SplitTest{faces, "~", -1, []string{faces}},
	SplitTest{faces, "", -1, []string{"☺", "☻", "☹"}},
	SplitTest{"1 2 3 4", " ", 3, []string{"1", "2", "3 4"}},
	SplitTest{"1 2", " ", 3, []string{"1", "2"}},
	SplitTest{"123", "", 2, []string{"1", "23"}},
	SplitTest{"123", "", 17, []string{"1", "2", "3"}},
}

func TestSplit(t *testing.T) {
	for _, tt := range splittests {
		a := Split([]byte(tt.s), []byte(tt.sep), tt.n)
		result := arrayOfString(a)
		if !eq(result, tt.a) {
			t.Errorf(`Split(%q, %q, %d) = %v; want %v`, tt.s, tt.sep, tt.n, result, tt.a)
			continue
		}
		if tt.n == 0 {
			continue
		}
		s := Join(a, []byte(tt.sep))
		if string(s) != tt.s {
			t.Errorf(`Join(Split(%q, %q, %d), %q) = %q`, tt.s, tt.sep, tt.n, tt.sep, s)
		}
	}
}

var splitaftertests = []SplitTest{
	SplitTest{abcd, "a", -1, []string{"a", "bcd"}},
	SplitTest{abcd, "z", -1, []string{"abcd"}},
	SplitTest{abcd, "", -1, []string{"a", "b", "c", "d"}},
	SplitTest{commas, ",", -1, []string{"1,", "2,", "3,", "4"}},
	SplitTest{dots, "...", -1, []string{"1...", ".2...", ".3...", ".4"}},
	SplitTest{faces, "☹", -1, []string{"☺☻☹", ""}},
	SplitTest{faces, "~", -1, []string{faces}},
	SplitTest{faces, "", -1, []string{"☺", "☻", "☹"}},
	SplitTest{"1 2 3 4", " ", 3, []string{"1 ", "2 ", "3 4"}},
	SplitTest{"1 2 3", " ", 3, []string{"1 ", "2 ", "3"}},
	SplitTest{"1 2", " ", 3, []string{"1 ", "2"}},
	SplitTest{"123", "", 2, []string{"1", "23"}},
	SplitTest{"123", "", 17, []string{"1", "2", "3"}},
}

func TestSplitAfter(t *testing.T) {
	for _, tt := range splitaftertests {
		a := SplitAfter([]byte(tt.s), []byte(tt.sep), tt.n)
		result := arrayOfString(a)
		if !eq(result, tt.a) {
			t.Errorf(`Split(%q, %q, %d) = %v; want %v`, tt.s, tt.sep, tt.n, result, tt.a)
			continue
		}
		s := Join(a, nil)
		if string(s) != tt.s {
			t.Errorf(`Join(Split(%q, %q, %d), %q) = %q`, tt.s, tt.sep, tt.n, tt.sep, s)
		}
	}
}

type FieldsTest struct {
	s string
	a []string
}

var fieldstests = []FieldsTest{
	FieldsTest{"", []string{}},
	FieldsTest{" ", []string{}},
	FieldsTest{" \t ", []string{}},
	FieldsTest{"  abc  ", []string{"abc"}},
	FieldsTest{"1 2 3 4", []string{"1", "2", "3", "4"}},
	FieldsTest{"1  2  3  4", []string{"1", "2", "3", "4"}},
	FieldsTest{"1\t\t2\t\t3\t4", []string{"1", "2", "3", "4"}},
	FieldsTest{"1\u20002\u20013\u20024", []string{"1", "2", "3", "4"}},
	FieldsTest{"\u2000\u2001\u2002", []string{}},
	FieldsTest{"\n™\t™\n", []string{"™", "™"}},
	FieldsTest{faces, []string{faces}},
}

func TestFields(t *testing.T) {
	for _, tt := range fieldstests {
		a := Fields([]byte(tt.s))
		result := arrayOfString(a)
		if !eq(result, tt.a) {
			t.Errorf("Fields(%q) = %v; want %v", tt.s, a, tt.a)
			continue
		}
	}
}

// Test case for any function which accepts and returns a byte array.
// For ease of creation, we write the byte arrays as strings.
type StringTest struct {
	in, out string
}

var upperTests = []StringTest{
	StringTest{"", ""},
	StringTest{"abc", "ABC"},
	StringTest{"AbC123", "ABC123"},
	StringTest{"azAZ09_", "AZAZ09_"},
	StringTest{"\u0250\u0250\u0250\u0250\u0250", "\u2C6F\u2C6F\u2C6F\u2C6F\u2C6F"}, // grows one byte per char
}

var lowerTests = []StringTest{
	StringTest{"", ""},
	StringTest{"abc", "abc"},
	StringTest{"AbC123", "abc123"},
	StringTest{"azAZ09_", "azaz09_"},
	StringTest{"\u2C6D\u2C6D\u2C6D\u2C6D\u2C6D", "\u0251\u0251\u0251\u0251\u0251"}, // shrinks one byte per char
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
	StringTest{" x\x80", "x\x80"},
	StringTest{" x\xc0", "x\xc0"},
	StringTest{"x \xc0\xc0 ", "x \xc0\xc0"},
	StringTest{"x \xc0", "x \xc0"},
	StringTest{"x \xc0 ", "x \xc0"},
	StringTest{"x \xc0\xc0 ", "x \xc0\xc0"},
	StringTest{"x ☺\xc0\xc0 ", "x ☺\xc0\xc0"},
	StringTest{"x ☺ ", "x ☺"},
}

// Bytes returns a new slice containing the bytes in s.
// Borrowed from strings to avoid dependency.
func Bytes(s string) []byte {
	b := make([]byte, len(s))
	for i := 0; i < len(s); i++ {
		b[i] = s[i]
	}
	return b
}

// Execute f on each test case.  funcName should be the name of f; it's used
// in failure reports.
func runStringTests(t *testing.T, f func([]byte) []byte, funcName string, testCases []StringTest) {
	for _, tc := range testCases {
		actual := string(f(Bytes(tc.in)))
		if actual != tc.out {
			t.Errorf("%s(%q) = %q; want %q", funcName, tc.in, actual, tc.out)
		}
	}
}

func tenRunes(rune int) string {
	r := make([]int, 10)
	for i := range r {
		r[i] = rune
	}
	return string(r)
}

// User-defined self-inverse mapping function
func rot13(rune int) int {
	step := 13
	if rune >= 'a' && rune <= 'z' {
		return ((rune - 'a' + step) % 26) + 'a'
	}
	if rune >= 'A' && rune <= 'Z' {
		return ((rune - 'A' + step) % 26) + 'A'
	}
	return rune
}

func TestMap(t *testing.T) {
	// Run a couple of awful growth/shrinkage tests
	a := tenRunes('a')

	// 1.  Grow.  This triggers two reallocations in Map.
	maxRune := func(rune int) int { return unicode.MaxRune }
	m := Map(maxRune, Bytes(a))
	expect := tenRunes(unicode.MaxRune)
	if string(m) != expect {
		t.Errorf("growing: expected %q got %q", expect, m)
	}

	// 2. Shrink
	minRune := func(rune int) int { return 'a' }
	m = Map(minRune, Bytes(tenRunes(unicode.MaxRune)))
	expect = a
	if string(m) != expect {
		t.Errorf("shrinking: expected %q got %q", expect, m)
	}

	// 3. Rot13
	m = Map(rot13, Bytes("a to zed"))
	expect = "n gb mrq"
	if string(m) != expect {
		t.Errorf("rot13: expected %q got %q", expect, m)
	}

	// 4. Rot13^2
	m = Map(rot13, Map(rot13, Bytes("a to zed")))
	expect = "a to zed"
	if string(m) != expect {
		t.Errorf("rot13: expected %q got %q", expect, m)
	}

	// 5. Drop
	dropNotLatin := func(rune int) int {
		if unicode.Is(unicode.Latin, rune) {
			return rune
		}
		return -1
	}
	m = Map(dropNotLatin, Bytes("Hello, 세계"))
	expect = "Hello"
	if string(m) != expect {
		t.Errorf("drop: expected %q got %q", expect, m)
	}
}

func TestToUpper(t *testing.T) { runStringTests(t, ToUpper, "ToUpper", upperTests) }

func TestToLower(t *testing.T) { runStringTests(t, ToLower, "ToLower", lowerTests) }

func TestTrimSpace(t *testing.T) { runStringTests(t, TrimSpace, "TrimSpace", trimSpaceTests) }

type AddTest struct {
	s, t string
	cap  int
}

var addtests = []AddTest{
	AddTest{"", "", 0},
	AddTest{"a", "", 1},
	AddTest{"a", "b", 1},
	AddTest{"abc", "def", 100},
}

func TestAdd(t *testing.T) {
	for _, test := range addtests {
		b := make([]byte, len(test.s), test.cap)
		for i := 0; i < len(test.s); i++ {
			b[i] = test.s[i]
		}
		b = Add(b, []byte(test.t))
		if string(b) != test.s+test.t {
			t.Errorf("Add(%q,%q) = %q", test.s, test.t, string(b))
		}
	}
}

func TestAddByte(t *testing.T) {
	const N = 2e5
	b := make([]byte, 0)
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
	in, out string
	count   int
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
		tin := []byte(tt.in)
		tout := []byte(tt.out)
		a := Repeat(tin, tt.count)
		if !Equal(a, tout) {
			t.Errorf("Repeat(%q, %d) = %q; want %q", tin, tt.count, a, tout)
			continue
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
	return true
}

type RunesTest struct {
	in    string
	out   []int
	lossy bool
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
		tin := []byte(tt.in)
		a := Runes(tin)
		if !runesEqual(a, tt.out) {
			t.Errorf("Runes(%q) = %v; want %v", tin, a, tt.out)
			continue
		}
		if !tt.lossy {
			// can only test reassembly if we didn't lose information
			s := string(a)
			if s != tt.in {
				t.Errorf("string(Runes(%q)) = %x; want %x", tin, s, tin)
			}
		}
	}
}


type TrimTest struct {
	f               func([]byte, string) []byte
	in, cutset, out string
}

var trimTests = []TrimTest{
	TrimTest{Trim, "abba", "a", "bb"},
	TrimTest{Trim, "abba", "ab", ""},
	TrimTest{TrimLeft, "abba", "ab", ""},
	TrimTest{TrimRight, "abba", "ab", ""},
	TrimTest{TrimLeft, "abba", "a", "bba"},
	TrimTest{TrimRight, "abba", "a", "abb"},
	TrimTest{Trim, "<tag>", "<>", "tag"},
	TrimTest{Trim, "* listitem", " *", "listitem"},
	TrimTest{Trim, `"quote"`, `"`, "quote"},
	TrimTest{Trim, "\u2C6F\u2C6F\u0250\u0250\u2C6F\u2C6F", "\u2C6F", "\u0250\u0250"},
	//empty string tests
	TrimTest{Trim, "abba", "", "abba"},
	TrimTest{Trim, "", "123", ""},
	TrimTest{Trim, "", "", ""},
	TrimTest{TrimLeft, "abba", "", "abba"},
	TrimTest{TrimLeft, "", "123", ""},
	TrimTest{TrimLeft, "", "", ""},
	TrimTest{TrimRight, "abba", "", "abba"},
	TrimTest{TrimRight, "", "123", ""},
	TrimTest{TrimRight, "", "", ""},
	TrimTest{TrimRight, "☺\xc0", "☺", "☺\xc0"},
}

func TestTrim(t *testing.T) {
	for _, tc := range trimTests {
		actual := string(tc.f([]byte(tc.in), tc.cutset))
		var name string
		switch tc.f {
		case Trim:
			name = "Trim"
		case TrimLeft:
			name = "TrimLeft"
		case TrimRight:
			name = "TrimRight"
		default:
			t.Error("Undefined trim function")
		}
		if actual != tc.out {
			t.Errorf("%s(%q, %q) = %q; want %q", name, tc.in, tc.cutset, actual, tc.out)
		}
	}
}

type predicate struct {
	f    func(r int) bool
	name string
}

var isSpace = predicate{unicode.IsSpace, "IsSpace"}
var isDigit = predicate{unicode.IsDigit, "IsDigit"}
var isUpper = predicate{unicode.IsUpper, "IsUpper"}
var isValidRune = predicate{
	func(r int) bool {
		return r != utf8.RuneError
	},
	"IsValidRune",
}

type TrimFuncTest struct {
	f       predicate
	in, out string
}

func not(p predicate) predicate {
	return predicate{
		func(r int) bool {
			return !p.f(r)
		},
		"not " + p.name,
	}
}

var trimFuncTests = []TrimFuncTest{
	TrimFuncTest{isSpace, space + " hello " + space, "hello"},
	TrimFuncTest{isDigit, "\u0e50\u0e5212hello34\u0e50\u0e51", "hello"},
	TrimFuncTest{isUpper, "\u2C6F\u2C6F\u2C6F\u2C6FABCDhelloEF\u2C6F\u2C6FGH\u2C6F\u2C6F", "hello"},
	TrimFuncTest{not(isSpace), "hello" + space + "hello", space},
	TrimFuncTest{not(isDigit), "hello\u0e50\u0e521234\u0e50\u0e51helo", "\u0e50\u0e521234\u0e50\u0e51"},
	TrimFuncTest{isValidRune, "ab\xc0a\xc0cd", "\xc0a\xc0"},
	TrimFuncTest{not(isValidRune), "\xc0a\xc0", "a"},
}

func TestTrimFunc(t *testing.T) {
	for _, tc := range trimFuncTests {
		actual := string(TrimFunc([]byte(tc.in), tc.f.f))
		if actual != tc.out {
			t.Errorf("TrimFunc(%q, %q) = %q; want %q", tc.in, tc.f.name, actual, tc.out)
		}
	}
}

type IndexFuncTest struct {
	in          string
	f           predicate
	first, last int
}

var indexFuncTests = []IndexFuncTest{
	IndexFuncTest{"", isValidRune, -1, -1},
	IndexFuncTest{"abc", isDigit, -1, -1},
	IndexFuncTest{"0123", isDigit, 0, 3},
	IndexFuncTest{"a1b", isDigit, 1, 1},
	IndexFuncTest{space, isSpace, 0, len(space) - 3}, // last rune in space is 3 bytes
	IndexFuncTest{"\u0e50\u0e5212hello34\u0e50\u0e51", isDigit, 0, 18},
	IndexFuncTest{"\u2C6F\u2C6F\u2C6F\u2C6FABCDhelloEF\u2C6F\u2C6FGH\u2C6F\u2C6F", isUpper, 0, 34},
	IndexFuncTest{"12\u0e50\u0e52hello34\u0e50\u0e51", not(isDigit), 8, 12},

	// tests of invalid UTF-8
	IndexFuncTest{"\x801", isDigit, 1, 1},
	IndexFuncTest{"\x80abc", isDigit, -1, -1},
	IndexFuncTest{"\xc0a\xc0", isValidRune, 1, 1},
	IndexFuncTest{"\xc0a\xc0", not(isValidRune), 0, 2},
	IndexFuncTest{"\xc0☺\xc0", not(isValidRune), 0, 4},
	IndexFuncTest{"\xc0☺\xc0\xc0", not(isValidRune), 0, 5},
	IndexFuncTest{"ab\xc0a\xc0cd", not(isValidRune), 2, 4},
	IndexFuncTest{"a\xe0\x80cd", not(isValidRune), 1, 2},
}

func TestIndexFunc(t *testing.T) {
	for _, tc := range indexFuncTests {
		first := IndexFunc([]byte(tc.in), tc.f.f)
		if first != tc.first {
			t.Errorf("IndexFunc(%q, %s) = %d; want %d", tc.in, tc.f.name, first, tc.first)
		}
		last := LastIndexFunc([]byte(tc.in), tc.f.f)
		if last != tc.last {
			t.Errorf("LastIndexFunc(%q, %s) = %d; want %d", tc.in, tc.f.name, last, tc.last)
		}
	}
}

type ReplaceTest struct {
	in       string
	old, new string
	n        int
	out      string
}

var ReplaceTests = []ReplaceTest{
	ReplaceTest{"hello", "l", "L", 0, "hello"},
	ReplaceTest{"hello", "l", "L", -1, "heLLo"},
	ReplaceTest{"hello", "x", "X", -1, "hello"},
	ReplaceTest{"", "x", "X", -1, ""},
	ReplaceTest{"radar", "r", "<r>", -1, "<r>ada<r>"},
	ReplaceTest{"", "", "<>", -1, "<>"},
	ReplaceTest{"banana", "a", "<>", -1, "b<>n<>n<>"},
	ReplaceTest{"banana", "a", "<>", 1, "b<>nana"},
	ReplaceTest{"banana", "a", "<>", 1000, "b<>n<>n<>"},
	ReplaceTest{"banana", "an", "<>", -1, "b<><>a"},
	ReplaceTest{"banana", "ana", "<>", -1, "b<>na"},
	ReplaceTest{"banana", "", "<>", -1, "<>b<>a<>n<>a<>n<>a<>"},
	ReplaceTest{"banana", "", "<>", 10, "<>b<>a<>n<>a<>n<>a<>"},
	ReplaceTest{"banana", "", "<>", 6, "<>b<>a<>n<>a<>n<>a"},
	ReplaceTest{"banana", "", "<>", 5, "<>b<>a<>n<>a<>na"},
	ReplaceTest{"banana", "", "<>", 1, "<>banana"},
	ReplaceTest{"banana", "a", "a", -1, "banana"},
	ReplaceTest{"banana", "a", "a", 1, "banana"},
	ReplaceTest{"☺☻☹", "", "<>", -1, "<>☺<>☻<>☹<>"},
}

func TestReplace(t *testing.T) {
	for _, tt := range ReplaceTests {
		if s := string(Replace([]byte(tt.in), []byte(tt.old), []byte(tt.new), tt.n)); s != tt.out {
			t.Errorf("Replace(%q, %q, %q, %d) = %q, want %q", tt.in, tt.old, tt.new, tt.n, s, tt.out)
		}
	}
}

type TitleTest struct {
	in, out string
}

var TitleTests = []TitleTest{
	TitleTest{"", ""},
	TitleTest{"a", "A"},
	TitleTest{" aaa aaa aaa ", " Aaa Aaa Aaa "},
	TitleTest{" Aaa Aaa Aaa ", " Aaa Aaa Aaa "},
	TitleTest{"123a456", "123a456"},
	TitleTest{"double-blind", "Double-Blind"},
	TitleTest{"ÿøû", "Ÿøû"},
}

func TestTitle(t *testing.T) {
	for _, tt := range TitleTests {
		if s := string(Title([]byte(tt.in))); s != tt.out {
			t.Errorf("Title(%q) = %q, want %q", tt.in, s, tt.out)
		}
	}
}
