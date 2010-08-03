// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strings_test

import (
	"os"
	. "strings"
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

var abcd = "abcd"
var faces = "☺☻☹"
var commas = "1,2,3,4"
var dots = "1....2....3....4"

type IndexTest struct {
	s   string
	sep string
	out int
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
	// cases with one byte strings - test special case in Index()
	IndexTest{"", "a", -1},
	IndexTest{"x", "a", -1},
	IndexTest{"x", "x", 0},
	IndexTest{"abc", "a", 0},
	IndexTest{"abc", "b", 1},
	IndexTest{"abc", "c", 2},
	IndexTest{"abc", "x", -1},
}

var lastIndexTests = []IndexTest{
	IndexTest{"", "", 0},
	IndexTest{"", "a", -1},
	IndexTest{"", "foo", -1},
	IndexTest{"fo", "foo", -1},
	IndexTest{"foo", "foo", 0},
	IndexTest{"foo", "f", 0},
	IndexTest{"oofofoofooo", "f", 7},
	IndexTest{"oofofoofooo", "foo", 7},
	IndexTest{"barfoobarfoo", "foo", 9},
	IndexTest{"foo", "", 3},
	IndexTest{"foo", "o", 2},
	IndexTest{"abcABCabc", "A", 3},
	IndexTest{"abcABCabc", "a", 6},
}

var indexAnyTests = []IndexTest{
	IndexTest{"", "", -1},
	IndexTest{"", "a", -1},
	IndexTest{"", "abc", -1},
	IndexTest{"a", "", -1},
	IndexTest{"a", "a", 0},
	IndexTest{"aaa", "a", 0},
	IndexTest{"abc", "xyz", -1},
	IndexTest{"abc", "xcz", 2},
	IndexTest{"a☺b☻c☹d", "uvw☻xyz", 2 + len("☺")},
	IndexTest{"aRegExp*", ".(|)*+?^$[]", 7},
	IndexTest{dots + dots + dots, " ", -1},
}

// Execute f on each test case.  funcName should be the name of f; it's used
// in failure reports.
func runIndexTests(t *testing.T, f func(s, sep string) int, funcName string, testCases []IndexTest) {
	for _, test := range testCases {
		actual := f(test.s, test.sep)
		if actual != test.out {
			t.Errorf("%s(%q,%q) = %v; want %v", funcName, test.s, test.sep, actual, test.out)
		}
	}
}

func TestIndex(t *testing.T)     { runIndexTests(t, Index, "Index", indexTests) }
func TestLastIndex(t *testing.T) { runIndexTests(t, LastIndex, "LastIndex", lastIndexTests) }
func TestIndexAny(t *testing.T)  { runIndexTests(t, IndexAny, "IndexAny", indexAnyTests) }

type ExplodeTest struct {
	s string
	n int
	a []string
}

var explodetests = []ExplodeTest{
	ExplodeTest{"", -1, []string{}},
	ExplodeTest{abcd, 4, []string{"a", "b", "c", "d"}},
	ExplodeTest{faces, 3, []string{"☺", "☻", "☹"}},
	ExplodeTest{abcd, 2, []string{"a", "bcd"}},
}

func TestExplode(t *testing.T) {
	for _, tt := range explodetests {
		a := Split(tt.s, "", tt.n)
		if !eq(a, tt.a) {
			t.Errorf("explode(%q, %d) = %v; want %v", tt.s, tt.n, a, tt.a)
			continue
		}
		s := Join(a, "")
		if s != tt.s {
			t.Errorf(`Join(explode(%q, %d), "") = %q`, tt.s, tt.n, s)
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
		a := Split(tt.s, tt.sep, tt.n)
		if !eq(a, tt.a) {
			t.Errorf("Split(%q, %q, %d) = %v; want %v", tt.s, tt.sep, tt.n, a, tt.a)
			continue
		}
		if tt.n == 0 {
			continue
		}
		s := Join(a, tt.sep)
		if s != tt.s {
			t.Errorf("Join(Split(%q, %q, %d), %q) = %q", tt.s, tt.sep, tt.n, tt.sep, s)
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
		a := SplitAfter(tt.s, tt.sep, tt.n)
		if !eq(a, tt.a) {
			t.Errorf(`Split(%q, %q, %d) = %v; want %v`, tt.s, tt.sep, tt.n, a, tt.a)
			continue
		}
		s := Join(a, "")
		if s != tt.s {
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
		a := Fields(tt.s)
		if !eq(a, tt.a) {
			t.Errorf("Fields(%q) = %v; want %v", tt.s, a, tt.a)
			continue
		}
	}
}

func TestFieldsFunc(t *testing.T) {
	pred := func(c int) bool { return c == 'X' }
	var fieldsFuncTests = []FieldsTest{
		FieldsTest{"", []string{}},
		FieldsTest{"XX", []string{}},
		FieldsTest{"XXhiXXX", []string{"hi"}},
		FieldsTest{"aXXbXXXcX", []string{"a", "b", "c"}},
	}
	for _, tt := range fieldsFuncTests {
		a := FieldsFunc(tt.s, pred)
		if !eq(a, tt.a) {
			t.Errorf("FieldsFunc(%q) = %v, want %v", tt.s, a, tt.a)
		}
	}
}


// Test case for any function which accepts and returns a single string.
type StringTest struct {
	in, out string
}

// Execute f on each test case.  funcName should be the name of f; it's used
// in failure reports.
func runStringTests(t *testing.T, f func(string) string, funcName string, testCases []StringTest) {
	for _, tc := range testCases {
		actual := f(tc.in)
		if actual != tc.out {
			t.Errorf("%s(%q) = %q; want %q", funcName, tc.in, actual, tc.out)
		}
	}
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
	m := Map(maxRune, a)
	expect := tenRunes(unicode.MaxRune)
	if m != expect {
		t.Errorf("growing: expected %q got %q", expect, m)
	}

	// 2. Shrink
	minRune := func(rune int) int { return 'a' }
	m = Map(minRune, tenRunes(unicode.MaxRune))
	expect = a
	if m != expect {
		t.Errorf("shrinking: expected %q got %q", expect, m)
	}

	// 3. Rot13
	m = Map(rot13, "a to zed")
	expect = "n gb mrq"
	if m != expect {
		t.Errorf("rot13: expected %q got %q", expect, m)
	}

	// 4. Rot13^2
	m = Map(rot13, Map(rot13, "a to zed"))
	expect = "a to zed"
	if m != expect {
		t.Errorf("rot13: expected %q got %q", expect, m)
	}

	// 5. Drop
	dropNotLatin := func(rune int) int {
		if unicode.Is(unicode.Latin, rune) {
			return rune
		}
		return -1
	}
	m = Map(dropNotLatin, "Hello, 세계")
	expect = "Hello"
	if m != expect {
		t.Errorf("drop: expected %q got %q", expect, m)
	}
}

func TestToUpper(t *testing.T) { runStringTests(t, ToUpper, "ToUpper", upperTests) }

func TestToLower(t *testing.T) { runStringTests(t, ToLower, "ToLower", lowerTests) }

func TestSpecialCase(t *testing.T) {
	lower := "abcçdefgğhıijklmnoöprsştuüvyz"
	upper := "ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ"
	u := ToUpperSpecial(unicode.TurkishCase, upper)
	if u != upper {
		t.Errorf("Upper(upper) is %s not %s", u, upper)
	}
	u = ToUpperSpecial(unicode.TurkishCase, lower)
	if u != upper {
		t.Errorf("Upper(lower) is %s not %s", u, upper)
	}
	l := ToLowerSpecial(unicode.TurkishCase, lower)
	if l != lower {
		t.Errorf("Lower(lower) is %s not %s", l, lower)
	}
	l = ToLowerSpecial(unicode.TurkishCase, upper)
	if l != lower {
		t.Errorf("Lower(upper) is %s not %s", l, lower)
	}
}

func TestTrimSpace(t *testing.T) { runStringTests(t, TrimSpace, "TrimSpace", trimSpaceTests) }

type TrimTest struct {
	f               func(string, string) string
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
		actual := tc.f(tc.in, tc.cutset)
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
		actual := TrimFunc(tc.in, tc.f.f)
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
		first := IndexFunc(tc.in, tc.f.f)
		if first != tc.first {
			t.Errorf("IndexFunc(%q, %s) = %d; want %d", tc.in, tc.f.name, first, tc.first)
		}
		last := LastIndexFunc(tc.in, tc.f.f)
		if last != tc.last {
			t.Errorf("LastIndexFunc(%q, %s) = %d; want %d", tc.in, tc.f.name, last, tc.last)
		}
	}
}

func equal(m string, s1, s2 string, t *testing.T) bool {
	if s1 == s2 {
		return true
	}
	e1 := Split(s1, "", -1)
	e2 := Split(s2, "", -1)
	for i, c1 := range e1 {
		if i > len(e2) {
			break
		}
		r1, _ := utf8.DecodeRuneInString(c1)
		r2, _ := utf8.DecodeRuneInString(e2[i])
		if r1 != r2 {
			t.Errorf("%s diff at %d: U+%04X U+%04X", m, i, r1, r2)
		}
	}
	return false
}

func TestCaseConsistency(t *testing.T) {
	// Make a string of all the runes.
	a := make([]int, unicode.MaxRune+1)
	for i := range a {
		a[i] = i
	}
	s := string(a)
	// convert the cases.
	upper := ToUpper(s)
	lower := ToLower(s)

	// Consistency checks
	if n := utf8.RuneCountInString(upper); n != unicode.MaxRune+1 {
		t.Error("rune count wrong in upper:", n)
	}
	if n := utf8.RuneCountInString(lower); n != unicode.MaxRune+1 {
		t.Error("rune count wrong in lower:", n)
	}
	if !equal("ToUpper(upper)", ToUpper(upper), upper, t) {
		t.Error("ToUpper(upper) consistency fail")
	}
	if !equal("ToLower(lower)", ToLower(lower), lower, t) {
		t.Error("ToLower(lower) consistency fail")
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
		a := Repeat(tt.in, tt.count)
		if !equal("Repeat(s)", a, tt.out, t) {
			t.Errorf("Repeat(%v, %d) = %v; want %v", tt.in, tt.count, a, tt.out)
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
		a := []int(tt.in)
		if !runesEqual(a, tt.out) {
			t.Errorf("[]int(%q) = %v; want %v", tt.in, a, tt.out)
			continue
		}
		if !tt.lossy {
			// can only test reassembly if we didn't lose information
			s := string(a)
			if s != tt.in {
				t.Errorf("string([]int(%q)) = %x; want %x", tt.in, s, tt.in)
			}
		}
	}
}

func TestReadRune(t *testing.T) {
	testStrings := []string{"", abcd, faces, commas}
	for _, s := range testStrings {
		reader := NewReader(s)
		res := ""
		for {
			r, _, e := reader.ReadRune()
			if e == os.EOF {
				break
			}
			if e != nil {
				t.Errorf("Reading %q: %s", s, e)
				break
			}
			res += string(r)
		}
		if res != s {
			t.Errorf("Reader(%q).ReadRune() produced %q", s, res)
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
		if s := Replace(tt.in, tt.old, tt.new, tt.n); s != tt.out {
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
		if s := Title(tt.in); s != tt.out {
			t.Errorf("Title(%q) = %q, want %q", tt.in, s, tt.out)
		}
	}
}
