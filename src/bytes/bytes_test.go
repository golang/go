// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bytes_test

import (
	. "bytes"
	"math/rand"
	"reflect"
	"testing"
	"unicode"
	"unicode/utf8"
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

func sliceOfString(s [][]byte) []string {
	result := make([]string, len(s))
	for i, v := range s {
		result[i] = string(v)
	}
	return result
}

// For ease of reading, the test cases use strings that are converted to byte
// slices before invoking the functions.

var abcd = "abcd"
var faces = "☺☻☹"
var commas = "1,2,3,4"
var dots = "1....2....3....4"

type BinOpTest struct {
	a string
	b string
	i int
}

var equalTests = []struct {
	a, b []byte
	i    int
}{
	{[]byte(""), []byte(""), 0},
	{[]byte("a"), []byte(""), 1},
	{[]byte(""), []byte("a"), -1},
	{[]byte("abc"), []byte("abc"), 0},
	{[]byte("ab"), []byte("abc"), -1},
	{[]byte("abc"), []byte("ab"), 1},
	{[]byte("x"), []byte("ab"), 1},
	{[]byte("ab"), []byte("x"), -1},
	{[]byte("x"), []byte("a"), 1},
	{[]byte("b"), []byte("x"), -1},
	// test runtime·memeq's chunked implementation
	{[]byte("abcdefgh"), []byte("abcdefgh"), 0},
	{[]byte("abcdefghi"), []byte("abcdefghi"), 0},
	{[]byte("abcdefghi"), []byte("abcdefghj"), -1},
	// nil tests
	{nil, nil, 0},
	{[]byte(""), nil, 0},
	{nil, []byte(""), 0},
	{[]byte("a"), nil, 1},
	{nil, []byte("a"), -1},
}

func TestEqual(t *testing.T) {
	for _, tt := range compareTests {
		eql := Equal(tt.a, tt.b)
		if eql != (tt.i == 0) {
			t.Errorf(`Equal(%q, %q) = %v`, tt.a, tt.b, eql)
		}
		eql = EqualPortable(tt.a, tt.b)
		if eql != (tt.i == 0) {
			t.Errorf(`EqualPortable(%q, %q) = %v`, tt.a, tt.b, eql)
		}
	}
}

func TestEqualExhaustive(t *testing.T) {
	var size = 128
	if testing.Short() {
		size = 32
	}
	a := make([]byte, size)
	b := make([]byte, size)
	b_init := make([]byte, size)
	// randomish but deterministic data
	for i := 0; i < size; i++ {
		a[i] = byte(17 * i)
		b_init[i] = byte(23*i + 100)
	}

	for len := 0; len <= size; len++ {
		for x := 0; x <= size-len; x++ {
			for y := 0; y <= size-len; y++ {
				copy(b, b_init)
				copy(b[y:y+len], a[x:x+len])
				if !Equal(a[x:x+len], b[y:y+len]) || !Equal(b[y:y+len], a[x:x+len]) {
					t.Errorf("Equal(%d, %d, %d) = false", len, x, y)
				}
			}
		}
	}
}

// make sure Equal returns false for minimally different strings.  The data
// is all zeros except for a single one in one location.
func TestNotEqual(t *testing.T) {
	var size = 128
	if testing.Short() {
		size = 32
	}
	a := make([]byte, size)
	b := make([]byte, size)

	for len := 0; len <= size; len++ {
		for x := 0; x <= size-len; x++ {
			for y := 0; y <= size-len; y++ {
				for diffpos := x; diffpos < x+len; diffpos++ {
					a[diffpos] = 1
					if Equal(a[x:x+len], b[y:y+len]) || Equal(b[y:y+len], a[x:x+len]) {
						t.Errorf("NotEqual(%d, %d, %d, %d) = true", len, x, y, diffpos)
					}
					a[diffpos] = 0
				}
			}
		}
	}
}

var indexTests = []BinOpTest{
	{"", "", 0},
	{"", "a", -1},
	{"", "foo", -1},
	{"fo", "foo", -1},
	{"foo", "baz", -1},
	{"foo", "foo", 0},
	{"oofofoofooo", "f", 2},
	{"oofofoofooo", "foo", 4},
	{"barfoobarfoo", "foo", 3},
	{"foo", "", 0},
	{"foo", "o", 1},
	{"abcABCabc", "A", 3},
	// cases with one byte strings - test IndexByte and special case in Index()
	{"", "a", -1},
	{"x", "a", -1},
	{"x", "x", 0},
	{"abc", "a", 0},
	{"abc", "b", 1},
	{"abc", "c", 2},
	{"abc", "x", -1},
	{"barfoobarfooyyyzzzyyyzzzyyyzzzyyyxxxzzzyyy", "x", 33},
	{"foofyfoobarfoobar", "y", 4},
	{"oooooooooooooooooooooo", "r", -1},
}

var lastIndexTests = []BinOpTest{
	{"", "", 0},
	{"", "a", -1},
	{"", "foo", -1},
	{"fo", "foo", -1},
	{"foo", "foo", 0},
	{"foo", "f", 0},
	{"oofofoofooo", "f", 7},
	{"oofofoofooo", "foo", 7},
	{"barfoobarfoo", "foo", 9},
	{"foo", "", 3},
	{"foo", "o", 2},
	{"abcABCabc", "A", 3},
	{"abcABCabc", "a", 6},
}

var indexAnyTests = []BinOpTest{
	{"", "", -1},
	{"", "a", -1},
	{"", "abc", -1},
	{"a", "", -1},
	{"a", "a", 0},
	{"aaa", "a", 0},
	{"abc", "xyz", -1},
	{"abc", "xcz", 2},
	{"ab☺c", "x☺yz", 2},
	{"aRegExp*", ".(|)*+?^$[]", 7},
	{dots + dots + dots, " ", -1},
}

var lastIndexAnyTests = []BinOpTest{
	{"", "", -1},
	{"", "a", -1},
	{"", "abc", -1},
	{"a", "", -1},
	{"a", "a", 0},
	{"aaa", "a", 2},
	{"abc", "xyz", -1},
	{"abc", "ab", 1},
	{"a☺b☻c☹d", "uvw☻xyz", 2 + len("☺")},
	{"a.RegExp*", ".(|)*+?^$[]", 8},
	{dots + dots + dots, " ", -1},
}

var indexRuneTests = []BinOpTest{
	{"", "a", -1},
	{"", "☺", -1},
	{"foo", "☹", -1},
	{"foo", "o", 1},
	{"foo☺bar", "☺", 3},
	{"foo☺☻☹bar", "☹", 9},
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

func runIndexAnyTests(t *testing.T, f func(s []byte, chars string) int, funcName string, testCases []BinOpTest) {
	for _, test := range testCases {
		a := []byte(test.a)
		actual := f(a, test.b)
		if actual != test.i {
			t.Errorf("%s(%q,%q) = %v; want %v", funcName, a, test.b, actual, test.i)
		}
	}
}

func TestIndex(t *testing.T)     { runIndexTests(t, Index, "Index", indexTests) }
func TestLastIndex(t *testing.T) { runIndexTests(t, LastIndex, "LastIndex", lastIndexTests) }
func TestIndexAny(t *testing.T)  { runIndexAnyTests(t, IndexAny, "IndexAny", indexAnyTests) }
func TestLastIndexAny(t *testing.T) {
	runIndexAnyTests(t, LastIndexAny, "LastIndexAny", lastIndexAnyTests)
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

func TestLastIndexByte(t *testing.T) {
	testCases := []BinOpTest{
		{"", "q", -1},
		{"abcdef", "q", -1},
		{"abcdefabcdef", "a", len("abcdef")},      // something in the middle
		{"abcdefabcdef", "f", len("abcdefabcde")}, // last byte
		{"zabcdefabcdef", "z", 0},                 // first byte
		{"a☺b☻c☹d", "b", len("a☺")},               // non-ascii
	}
	for _, test := range testCases {
		actual := LastIndexByte([]byte(test.a), test.b[0])
		if actual != test.i {
			t.Errorf("LastIndexByte(%q,%c) = %v; want %v", test.a, test.b[0], actual, test.i)
		}
	}
}

// test a larger buffer with different sizes and alignments
func TestIndexByteBig(t *testing.T) {
	var n = 1024
	if testing.Short() {
		n = 128
	}
	b := make([]byte, n)
	for i := 0; i < n; i++ {
		// different start alignments
		b1 := b[i:]
		for j := 0; j < len(b1); j++ {
			b1[j] = 'x'
			pos := IndexByte(b1, 'x')
			if pos != j {
				t.Errorf("IndexByte(%q, 'x') = %v", b1, pos)
			}
			b1[j] = 0
			pos = IndexByte(b1, 'x')
			if pos != -1 {
				t.Errorf("IndexByte(%q, 'x') = %v", b1, pos)
			}
		}
		// different end alignments
		b1 = b[:i]
		for j := 0; j < len(b1); j++ {
			b1[j] = 'x'
			pos := IndexByte(b1, 'x')
			if pos != j {
				t.Errorf("IndexByte(%q, 'x') = %v", b1, pos)
			}
			b1[j] = 0
			pos = IndexByte(b1, 'x')
			if pos != -1 {
				t.Errorf("IndexByte(%q, 'x') = %v", b1, pos)
			}
		}
		// different start and end alignments
		b1 = b[i/2 : n-(i+1)/2]
		for j := 0; j < len(b1); j++ {
			b1[j] = 'x'
			pos := IndexByte(b1, 'x')
			if pos != j {
				t.Errorf("IndexByte(%q, 'x') = %v", b1, pos)
			}
			b1[j] = 0
			pos = IndexByte(b1, 'x')
			if pos != -1 {
				t.Errorf("IndexByte(%q, 'x') = %v", b1, pos)
			}
		}
	}
}

func TestIndexRune(t *testing.T) {
	for _, tt := range indexRuneTests {
		a := []byte(tt.a)
		r, _ := utf8.DecodeRuneInString(tt.b)
		pos := IndexRune(a, r)
		if pos != tt.i {
			t.Errorf(`IndexRune(%q, '%c') = %v`, tt.a, r, pos)
		}
	}
}

var bmbuf []byte

func BenchmarkIndexByte32(b *testing.B)          { bmIndexByte(b, IndexByte, 32) }
func BenchmarkIndexByte4K(b *testing.B)          { bmIndexByte(b, IndexByte, 4<<10) }
func BenchmarkIndexByte4M(b *testing.B)          { bmIndexByte(b, IndexByte, 4<<20) }
func BenchmarkIndexByte64M(b *testing.B)         { bmIndexByte(b, IndexByte, 64<<20) }
func BenchmarkIndexBytePortable32(b *testing.B)  { bmIndexByte(b, IndexBytePortable, 32) }
func BenchmarkIndexBytePortable4K(b *testing.B)  { bmIndexByte(b, IndexBytePortable, 4<<10) }
func BenchmarkIndexBytePortable4M(b *testing.B)  { bmIndexByte(b, IndexBytePortable, 4<<20) }
func BenchmarkIndexBytePortable64M(b *testing.B) { bmIndexByte(b, IndexBytePortable, 64<<20) }

func bmIndexByte(b *testing.B, index func([]byte, byte) int, n int) {
	if len(bmbuf) < n {
		bmbuf = make([]byte, n)
	}
	b.SetBytes(int64(n))
	buf := bmbuf[0:n]
	buf[n-1] = 'x'
	for i := 0; i < b.N; i++ {
		j := index(buf, 'x')
		if j != n-1 {
			b.Fatal("bad index", j)
		}
	}
	buf[n-1] = '\x00'
}

func BenchmarkEqual0(b *testing.B) {
	var buf [4]byte
	buf1 := buf[0:0]
	buf2 := buf[1:1]
	for i := 0; i < b.N; i++ {
		eq := Equal(buf1, buf2)
		if !eq {
			b.Fatal("bad equal")
		}
	}
}

func BenchmarkEqual1(b *testing.B)           { bmEqual(b, Equal, 1) }
func BenchmarkEqual6(b *testing.B)           { bmEqual(b, Equal, 6) }
func BenchmarkEqual9(b *testing.B)           { bmEqual(b, Equal, 9) }
func BenchmarkEqual15(b *testing.B)          { bmEqual(b, Equal, 15) }
func BenchmarkEqual16(b *testing.B)          { bmEqual(b, Equal, 16) }
func BenchmarkEqual20(b *testing.B)          { bmEqual(b, Equal, 20) }
func BenchmarkEqual32(b *testing.B)          { bmEqual(b, Equal, 32) }
func BenchmarkEqual4K(b *testing.B)          { bmEqual(b, Equal, 4<<10) }
func BenchmarkEqual4M(b *testing.B)          { bmEqual(b, Equal, 4<<20) }
func BenchmarkEqual64M(b *testing.B)         { bmEqual(b, Equal, 64<<20) }
func BenchmarkEqualPort1(b *testing.B)       { bmEqual(b, EqualPortable, 1) }
func BenchmarkEqualPort6(b *testing.B)       { bmEqual(b, EqualPortable, 6) }
func BenchmarkEqualPort32(b *testing.B)      { bmEqual(b, EqualPortable, 32) }
func BenchmarkEqualPort4K(b *testing.B)      { bmEqual(b, EqualPortable, 4<<10) }
func BenchmarkEqualPortable4M(b *testing.B)  { bmEqual(b, EqualPortable, 4<<20) }
func BenchmarkEqualPortable64M(b *testing.B) { bmEqual(b, EqualPortable, 64<<20) }

func bmEqual(b *testing.B, equal func([]byte, []byte) bool, n int) {
	if len(bmbuf) < 2*n {
		bmbuf = make([]byte, 2*n)
	}
	b.SetBytes(int64(n))
	buf1 := bmbuf[0:n]
	buf2 := bmbuf[n : 2*n]
	buf1[n-1] = 'x'
	buf2[n-1] = 'x'
	for i := 0; i < b.N; i++ {
		eq := equal(buf1, buf2)
		if !eq {
			b.Fatal("bad equal")
		}
	}
	buf1[n-1] = '\x00'
	buf2[n-1] = '\x00'
}

func BenchmarkIndex32(b *testing.B)  { bmIndex(b, Index, 32) }
func BenchmarkIndex4K(b *testing.B)  { bmIndex(b, Index, 4<<10) }
func BenchmarkIndex4M(b *testing.B)  { bmIndex(b, Index, 4<<20) }
func BenchmarkIndex64M(b *testing.B) { bmIndex(b, Index, 64<<20) }

func bmIndex(b *testing.B, index func([]byte, []byte) int, n int) {
	if len(bmbuf) < n {
		bmbuf = make([]byte, n)
	}
	b.SetBytes(int64(n))
	buf := bmbuf[0:n]
	buf[n-1] = 'x'
	for i := 0; i < b.N; i++ {
		j := index(buf, buf[n-7:])
		if j != n-7 {
			b.Fatal("bad index", j)
		}
	}
	buf[n-1] = '\x00'
}

func BenchmarkIndexEasy32(b *testing.B)  { bmIndexEasy(b, Index, 32) }
func BenchmarkIndexEasy4K(b *testing.B)  { bmIndexEasy(b, Index, 4<<10) }
func BenchmarkIndexEasy4M(b *testing.B)  { bmIndexEasy(b, Index, 4<<20) }
func BenchmarkIndexEasy64M(b *testing.B) { bmIndexEasy(b, Index, 64<<20) }

func bmIndexEasy(b *testing.B, index func([]byte, []byte) int, n int) {
	if len(bmbuf) < n {
		bmbuf = make([]byte, n)
	}
	b.SetBytes(int64(n))
	buf := bmbuf[0:n]
	buf[n-1] = 'x'
	buf[n-7] = 'x'
	for i := 0; i < b.N; i++ {
		j := index(buf, buf[n-7:])
		if j != n-7 {
			b.Fatal("bad index", j)
		}
	}
	buf[n-1] = '\x00'
	buf[n-7] = '\x00'
}

func BenchmarkCount32(b *testing.B)  { bmCount(b, Count, 32) }
func BenchmarkCount4K(b *testing.B)  { bmCount(b, Count, 4<<10) }
func BenchmarkCount4M(b *testing.B)  { bmCount(b, Count, 4<<20) }
func BenchmarkCount64M(b *testing.B) { bmCount(b, Count, 64<<20) }

func bmCount(b *testing.B, count func([]byte, []byte) int, n int) {
	if len(bmbuf) < n {
		bmbuf = make([]byte, n)
	}
	b.SetBytes(int64(n))
	buf := bmbuf[0:n]
	buf[n-1] = 'x'
	for i := 0; i < b.N; i++ {
		j := count(buf, buf[n-7:])
		if j != 1 {
			b.Fatal("bad count", j)
		}
	}
	buf[n-1] = '\x00'
}

func BenchmarkCountEasy32(b *testing.B)  { bmCountEasy(b, Count, 32) }
func BenchmarkCountEasy4K(b *testing.B)  { bmCountEasy(b, Count, 4<<10) }
func BenchmarkCountEasy4M(b *testing.B)  { bmCountEasy(b, Count, 4<<20) }
func BenchmarkCountEasy64M(b *testing.B) { bmCountEasy(b, Count, 64<<20) }

func bmCountEasy(b *testing.B, count func([]byte, []byte) int, n int) {
	if len(bmbuf) < n {
		bmbuf = make([]byte, n)
	}
	b.SetBytes(int64(n))
	buf := bmbuf[0:n]
	buf[n-1] = 'x'
	buf[n-7] = 'x'
	for i := 0; i < b.N; i++ {
		j := count(buf, buf[n-7:])
		if j != 1 {
			b.Fatal("bad count", j)
		}
	}
	buf[n-1] = '\x00'
	buf[n-7] = '\x00'
}

type ExplodeTest struct {
	s string
	n int
	a []string
}

var explodetests = []ExplodeTest{
	{"", -1, []string{}},
	{abcd, -1, []string{"a", "b", "c", "d"}},
	{faces, -1, []string{"☺", "☻", "☹"}},
	{abcd, 2, []string{"a", "bcd"}},
}

func TestExplode(t *testing.T) {
	for _, tt := range explodetests {
		a := SplitN([]byte(tt.s), nil, tt.n)
		result := sliceOfString(a)
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
	{abcd, "a", 0, nil},
	{abcd, "a", -1, []string{"", "bcd"}},
	{abcd, "z", -1, []string{"abcd"}},
	{abcd, "", -1, []string{"a", "b", "c", "d"}},
	{commas, ",", -1, []string{"1", "2", "3", "4"}},
	{dots, "...", -1, []string{"1", ".2", ".3", ".4"}},
	{faces, "☹", -1, []string{"☺☻", ""}},
	{faces, "~", -1, []string{faces}},
	{faces, "", -1, []string{"☺", "☻", "☹"}},
	{"1 2 3 4", " ", 3, []string{"1", "2", "3 4"}},
	{"1 2", " ", 3, []string{"1", "2"}},
	{"123", "", 2, []string{"1", "23"}},
	{"123", "", 17, []string{"1", "2", "3"}},
}

func TestSplit(t *testing.T) {
	for _, tt := range splittests {
		a := SplitN([]byte(tt.s), []byte(tt.sep), tt.n)
		result := sliceOfString(a)
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
		if tt.n < 0 {
			b := Split([]byte(tt.s), []byte(tt.sep))
			if !reflect.DeepEqual(a, b) {
				t.Errorf("Split disagrees withSplitN(%q, %q, %d) = %v; want %v", tt.s, tt.sep, tt.n, b, a)
			}
		}
		if len(a) > 0 {
			in, out := a[0], s
			if cap(in) == cap(out) && &in[:1][0] == &out[:1][0] {
				t.Errorf("Join(%#v, %q) didn't copy", a, tt.sep)
			}
		}
	}
}

var splitaftertests = []SplitTest{
	{abcd, "a", -1, []string{"a", "bcd"}},
	{abcd, "z", -1, []string{"abcd"}},
	{abcd, "", -1, []string{"a", "b", "c", "d"}},
	{commas, ",", -1, []string{"1,", "2,", "3,", "4"}},
	{dots, "...", -1, []string{"1...", ".2...", ".3...", ".4"}},
	{faces, "☹", -1, []string{"☺☻☹", ""}},
	{faces, "~", -1, []string{faces}},
	{faces, "", -1, []string{"☺", "☻", "☹"}},
	{"1 2 3 4", " ", 3, []string{"1 ", "2 ", "3 4"}},
	{"1 2 3", " ", 3, []string{"1 ", "2 ", "3"}},
	{"1 2", " ", 3, []string{"1 ", "2"}},
	{"123", "", 2, []string{"1", "23"}},
	{"123", "", 17, []string{"1", "2", "3"}},
}

func TestSplitAfter(t *testing.T) {
	for _, tt := range splitaftertests {
		a := SplitAfterN([]byte(tt.s), []byte(tt.sep), tt.n)
		result := sliceOfString(a)
		if !eq(result, tt.a) {
			t.Errorf(`Split(%q, %q, %d) = %v; want %v`, tt.s, tt.sep, tt.n, result, tt.a)
			continue
		}
		s := Join(a, nil)
		if string(s) != tt.s {
			t.Errorf(`Join(Split(%q, %q, %d), %q) = %q`, tt.s, tt.sep, tt.n, tt.sep, s)
		}
		if tt.n < 0 {
			b := SplitAfter([]byte(tt.s), []byte(tt.sep))
			if !reflect.DeepEqual(a, b) {
				t.Errorf("SplitAfter disagrees withSplitAfterN(%q, %q, %d) = %v; want %v", tt.s, tt.sep, tt.n, b, a)
			}
		}
	}
}

type FieldsTest struct {
	s string
	a []string
}

var fieldstests = []FieldsTest{
	{"", []string{}},
	{" ", []string{}},
	{" \t ", []string{}},
	{"  abc  ", []string{"abc"}},
	{"1 2 3 4", []string{"1", "2", "3", "4"}},
	{"1  2  3  4", []string{"1", "2", "3", "4"}},
	{"1\t\t2\t\t3\t4", []string{"1", "2", "3", "4"}},
	{"1\u20002\u20013\u20024", []string{"1", "2", "3", "4"}},
	{"\u2000\u2001\u2002", []string{}},
	{"\n™\t™\n", []string{"™", "™"}},
	{faces, []string{faces}},
}

func TestFields(t *testing.T) {
	for _, tt := range fieldstests {
		a := Fields([]byte(tt.s))
		result := sliceOfString(a)
		if !eq(result, tt.a) {
			t.Errorf("Fields(%q) = %v; want %v", tt.s, a, tt.a)
			continue
		}
	}
}

func TestFieldsFunc(t *testing.T) {
	for _, tt := range fieldstests {
		a := FieldsFunc([]byte(tt.s), unicode.IsSpace)
		result := sliceOfString(a)
		if !eq(result, tt.a) {
			t.Errorf("FieldsFunc(%q, unicode.IsSpace) = %v; want %v", tt.s, a, tt.a)
			continue
		}
	}
	pred := func(c rune) bool { return c == 'X' }
	var fieldsFuncTests = []FieldsTest{
		{"", []string{}},
		{"XX", []string{}},
		{"XXhiXXX", []string{"hi"}},
		{"aXXbXXXcX", []string{"a", "b", "c"}},
	}
	for _, tt := range fieldsFuncTests {
		a := FieldsFunc([]byte(tt.s), pred)
		result := sliceOfString(a)
		if !eq(result, tt.a) {
			t.Errorf("FieldsFunc(%q) = %v, want %v", tt.s, a, tt.a)
		}
	}
}

// Test case for any function which accepts and returns a byte slice.
// For ease of creation, we write the byte slices as strings.
type StringTest struct {
	in, out string
}

var upperTests = []StringTest{
	{"", ""},
	{"abc", "ABC"},
	{"AbC123", "ABC123"},
	{"azAZ09_", "AZAZ09_"},
	{"\u0250\u0250\u0250\u0250\u0250", "\u2C6F\u2C6F\u2C6F\u2C6F\u2C6F"}, // grows one byte per char
}

var lowerTests = []StringTest{
	{"", ""},
	{"abc", "abc"},
	{"AbC123", "abc123"},
	{"azAZ09_", "azaz09_"},
	{"\u2C6D\u2C6D\u2C6D\u2C6D\u2C6D", "\u0251\u0251\u0251\u0251\u0251"}, // shrinks one byte per char
}

const space = "\t\v\r\f\n\u0085\u00a0\u2000\u3000"

var trimSpaceTests = []StringTest{
	{"", ""},
	{"abc", "abc"},
	{space + "abc" + space, "abc"},
	{" ", ""},
	{" \t\r\n \t\t\r\r\n\n ", ""},
	{" \t\r\n x\t\t\r\r\n\n ", "x"},
	{" \u2000\t\r\n x\t\t\r\r\ny\n \u3000", "x\t\t\r\r\ny"},
	{"1 \t\r\n2", "1 \t\r\n2"},
	{" x\x80", "x\x80"},
	{" x\xc0", "x\xc0"},
	{"x \xc0\xc0 ", "x \xc0\xc0"},
	{"x \xc0", "x \xc0"},
	{"x \xc0 ", "x \xc0"},
	{"x \xc0\xc0 ", "x \xc0\xc0"},
	{"x ☺\xc0\xc0 ", "x ☺\xc0\xc0"},
	{"x ☺ ", "x ☺"},
}

// Execute f on each test case.  funcName should be the name of f; it's used
// in failure reports.
func runStringTests(t *testing.T, f func([]byte) []byte, funcName string, testCases []StringTest) {
	for _, tc := range testCases {
		actual := string(f([]byte(tc.in)))
		if actual != tc.out {
			t.Errorf("%s(%q) = %q; want %q", funcName, tc.in, actual, tc.out)
		}
	}
}

func tenRunes(r rune) string {
	runes := make([]rune, 10)
	for i := range runes {
		runes[i] = r
	}
	return string(runes)
}

// User-defined self-inverse mapping function
func rot13(r rune) rune {
	const step = 13
	if r >= 'a' && r <= 'z' {
		return ((r - 'a' + step) % 26) + 'a'
	}
	if r >= 'A' && r <= 'Z' {
		return ((r - 'A' + step) % 26) + 'A'
	}
	return r
}

func TestMap(t *testing.T) {
	// Run a couple of awful growth/shrinkage tests
	a := tenRunes('a')

	// 1.  Grow.  This triggers two reallocations in Map.
	maxRune := func(r rune) rune { return unicode.MaxRune }
	m := Map(maxRune, []byte(a))
	expect := tenRunes(unicode.MaxRune)
	if string(m) != expect {
		t.Errorf("growing: expected %q got %q", expect, m)
	}

	// 2. Shrink
	minRune := func(r rune) rune { return 'a' }
	m = Map(minRune, []byte(tenRunes(unicode.MaxRune)))
	expect = a
	if string(m) != expect {
		t.Errorf("shrinking: expected %q got %q", expect, m)
	}

	// 3. Rot13
	m = Map(rot13, []byte("a to zed"))
	expect = "n gb mrq"
	if string(m) != expect {
		t.Errorf("rot13: expected %q got %q", expect, m)
	}

	// 4. Rot13^2
	m = Map(rot13, Map(rot13, []byte("a to zed")))
	expect = "a to zed"
	if string(m) != expect {
		t.Errorf("rot13: expected %q got %q", expect, m)
	}

	// 5. Drop
	dropNotLatin := func(r rune) rune {
		if unicode.Is(unicode.Latin, r) {
			return r
		}
		return -1
	}
	m = Map(dropNotLatin, []byte("Hello, 세계"))
	expect = "Hello"
	if string(m) != expect {
		t.Errorf("drop: expected %q got %q", expect, m)
	}

	// 6. Invalid rune
	invalidRune := func(r rune) rune {
		return utf8.MaxRune + 1
	}
	m = Map(invalidRune, []byte("x"))
	expect = "\uFFFD"
	if string(m) != expect {
		t.Errorf("invalidRune: expected %q got %q", expect, m)
	}
}

func TestToUpper(t *testing.T) { runStringTests(t, ToUpper, "ToUpper", upperTests) }

func TestToLower(t *testing.T) { runStringTests(t, ToLower, "ToLower", lowerTests) }

func TestTrimSpace(t *testing.T) { runStringTests(t, TrimSpace, "TrimSpace", trimSpaceTests) }

type RepeatTest struct {
	in, out string
	count   int
}

var RepeatTests = []RepeatTest{
	{"", "", 0},
	{"", "", 1},
	{"", "", 2},
	{"-", "", 0},
	{"-", "-", 1},
	{"-", "----------", 10},
	{"abc ", "abc abc abc ", 3},
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

func runesEqual(a, b []rune) bool {
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
	out   []rune
	lossy bool
}

var RunesTests = []RunesTest{
	{"", []rune{}, false},
	{" ", []rune{32}, false},
	{"ABC", []rune{65, 66, 67}, false},
	{"abc", []rune{97, 98, 99}, false},
	{"\u65e5\u672c\u8a9e", []rune{26085, 26412, 35486}, false},
	{"ab\x80c", []rune{97, 98, 0xFFFD, 99}, true},
	{"ab\xc0c", []rune{97, 98, 0xFFFD, 99}, true},
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
	f            string
	in, arg, out string
}

var trimTests = []TrimTest{
	{"Trim", "abba", "a", "bb"},
	{"Trim", "abba", "ab", ""},
	{"TrimLeft", "abba", "ab", ""},
	{"TrimRight", "abba", "ab", ""},
	{"TrimLeft", "abba", "a", "bba"},
	{"TrimRight", "abba", "a", "abb"},
	{"Trim", "<tag>", "<>", "tag"},
	{"Trim", "* listitem", " *", "listitem"},
	{"Trim", `"quote"`, `"`, "quote"},
	{"Trim", "\u2C6F\u2C6F\u0250\u0250\u2C6F\u2C6F", "\u2C6F", "\u0250\u0250"},
	//empty string tests
	{"Trim", "abba", "", "abba"},
	{"Trim", "", "123", ""},
	{"Trim", "", "", ""},
	{"TrimLeft", "abba", "", "abba"},
	{"TrimLeft", "", "123", ""},
	{"TrimLeft", "", "", ""},
	{"TrimRight", "abba", "", "abba"},
	{"TrimRight", "", "123", ""},
	{"TrimRight", "", "", ""},
	{"TrimRight", "☺\xc0", "☺", "☺\xc0"},
	{"TrimPrefix", "aabb", "a", "abb"},
	{"TrimPrefix", "aabb", "b", "aabb"},
	{"TrimSuffix", "aabb", "a", "aabb"},
	{"TrimSuffix", "aabb", "b", "aab"},
}

func TestTrim(t *testing.T) {
	for _, tc := range trimTests {
		name := tc.f
		var f func([]byte, string) []byte
		var fb func([]byte, []byte) []byte
		switch name {
		case "Trim":
			f = Trim
		case "TrimLeft":
			f = TrimLeft
		case "TrimRight":
			f = TrimRight
		case "TrimPrefix":
			fb = TrimPrefix
		case "TrimSuffix":
			fb = TrimSuffix
		default:
			t.Errorf("Undefined trim function %s", name)
		}
		var actual string
		if f != nil {
			actual = string(f([]byte(tc.in), tc.arg))
		} else {
			actual = string(fb([]byte(tc.in), []byte(tc.arg)))
		}
		if actual != tc.out {
			t.Errorf("%s(%q, %q) = %q; want %q", name, tc.in, tc.arg, actual, tc.out)
		}
	}
}

type predicate struct {
	f    func(r rune) bool
	name string
}

var isSpace = predicate{unicode.IsSpace, "IsSpace"}
var isDigit = predicate{unicode.IsDigit, "IsDigit"}
var isUpper = predicate{unicode.IsUpper, "IsUpper"}
var isValidRune = predicate{
	func(r rune) bool {
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
		func(r rune) bool {
			return !p.f(r)
		},
		"not " + p.name,
	}
}

var trimFuncTests = []TrimFuncTest{
	{isSpace, space + " hello " + space, "hello"},
	{isDigit, "\u0e50\u0e5212hello34\u0e50\u0e51", "hello"},
	{isUpper, "\u2C6F\u2C6F\u2C6F\u2C6FABCDhelloEF\u2C6F\u2C6FGH\u2C6F\u2C6F", "hello"},
	{not(isSpace), "hello" + space + "hello", space},
	{not(isDigit), "hello\u0e50\u0e521234\u0e50\u0e51helo", "\u0e50\u0e521234\u0e50\u0e51"},
	{isValidRune, "ab\xc0a\xc0cd", "\xc0a\xc0"},
	{not(isValidRune), "\xc0a\xc0", "a"},
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
	{"", isValidRune, -1, -1},
	{"abc", isDigit, -1, -1},
	{"0123", isDigit, 0, 3},
	{"a1b", isDigit, 1, 1},
	{space, isSpace, 0, len(space) - 3}, // last rune in space is 3 bytes
	{"\u0e50\u0e5212hello34\u0e50\u0e51", isDigit, 0, 18},
	{"\u2C6F\u2C6F\u2C6F\u2C6FABCDhelloEF\u2C6F\u2C6FGH\u2C6F\u2C6F", isUpper, 0, 34},
	{"12\u0e50\u0e52hello34\u0e50\u0e51", not(isDigit), 8, 12},

	// tests of invalid UTF-8
	{"\x801", isDigit, 1, 1},
	{"\x80abc", isDigit, -1, -1},
	{"\xc0a\xc0", isValidRune, 1, 1},
	{"\xc0a\xc0", not(isValidRune), 0, 2},
	{"\xc0☺\xc0", not(isValidRune), 0, 4},
	{"\xc0☺\xc0\xc0", not(isValidRune), 0, 5},
	{"ab\xc0a\xc0cd", not(isValidRune), 2, 4},
	{"a\xe0\x80cd", not(isValidRune), 1, 2},
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
	{"hello", "l", "L", 0, "hello"},
	{"hello", "l", "L", -1, "heLLo"},
	{"hello", "x", "X", -1, "hello"},
	{"", "x", "X", -1, ""},
	{"radar", "r", "<r>", -1, "<r>ada<r>"},
	{"", "", "<>", -1, "<>"},
	{"banana", "a", "<>", -1, "b<>n<>n<>"},
	{"banana", "a", "<>", 1, "b<>nana"},
	{"banana", "a", "<>", 1000, "b<>n<>n<>"},
	{"banana", "an", "<>", -1, "b<><>a"},
	{"banana", "ana", "<>", -1, "b<>na"},
	{"banana", "", "<>", -1, "<>b<>a<>n<>a<>n<>a<>"},
	{"banana", "", "<>", 10, "<>b<>a<>n<>a<>n<>a<>"},
	{"banana", "", "<>", 6, "<>b<>a<>n<>a<>n<>a"},
	{"banana", "", "<>", 5, "<>b<>a<>n<>a<>na"},
	{"banana", "", "<>", 1, "<>banana"},
	{"banana", "a", "a", -1, "banana"},
	{"banana", "a", "a", 1, "banana"},
	{"☺☻☹", "", "<>", -1, "<>☺<>☻<>☹<>"},
}

func TestReplace(t *testing.T) {
	for _, tt := range ReplaceTests {
		in := append([]byte(tt.in), "<spare>"...)
		in = in[:len(tt.in)]
		out := Replace(in, []byte(tt.old), []byte(tt.new), tt.n)
		if s := string(out); s != tt.out {
			t.Errorf("Replace(%q, %q, %q, %d) = %q, want %q", tt.in, tt.old, tt.new, tt.n, s, tt.out)
		}
		if cap(in) == cap(out) && &in[:1][0] == &out[:1][0] {
			t.Errorf("Replace(%q, %q, %q, %d) didn't copy", tt.in, tt.old, tt.new, tt.n)
		}
	}
}

type TitleTest struct {
	in, out string
}

var TitleTests = []TitleTest{
	{"", ""},
	{"a", "A"},
	{" aaa aaa aaa ", " Aaa Aaa Aaa "},
	{" Aaa Aaa Aaa ", " Aaa Aaa Aaa "},
	{"123a456", "123a456"},
	{"double-blind", "Double-Blind"},
	{"ÿøû", "Ÿøû"},
	{"with_underscore", "With_underscore"},
	{"unicode \xe2\x80\xa8 line separator", "Unicode \xe2\x80\xa8 Line Separator"},
}

func TestTitle(t *testing.T) {
	for _, tt := range TitleTests {
		if s := string(Title([]byte(tt.in))); s != tt.out {
			t.Errorf("Title(%q) = %q, want %q", tt.in, s, tt.out)
		}
	}
}

var ToTitleTests = []TitleTest{
	{"", ""},
	{"a", "A"},
	{" aaa aaa aaa ", " AAA AAA AAA "},
	{" Aaa Aaa Aaa ", " AAA AAA AAA "},
	{"123a456", "123A456"},
	{"double-blind", "DOUBLE-BLIND"},
	{"ÿøû", "ŸØÛ"},
}

func TestToTitle(t *testing.T) {
	for _, tt := range ToTitleTests {
		if s := string(ToTitle([]byte(tt.in))); s != tt.out {
			t.Errorf("ToTitle(%q) = %q, want %q", tt.in, s, tt.out)
		}
	}
}

var EqualFoldTests = []struct {
	s, t string
	out  bool
}{
	{"abc", "abc", true},
	{"ABcd", "ABcd", true},
	{"123abc", "123ABC", true},
	{"αβδ", "ΑΒΔ", true},
	{"abc", "xyz", false},
	{"abc", "XYZ", false},
	{"abcdefghijk", "abcdefghijX", false},
	{"abcdefghijk", "abcdefghij\u212A", true},
	{"abcdefghijK", "abcdefghij\u212A", true},
	{"abcdefghijkz", "abcdefghij\u212Ay", false},
	{"abcdefghijKz", "abcdefghij\u212Ay", false},
}

func TestEqualFold(t *testing.T) {
	for _, tt := range EqualFoldTests {
		if out := EqualFold([]byte(tt.s), []byte(tt.t)); out != tt.out {
			t.Errorf("EqualFold(%#q, %#q) = %v, want %v", tt.s, tt.t, out, tt.out)
		}
		if out := EqualFold([]byte(tt.t), []byte(tt.s)); out != tt.out {
			t.Errorf("EqualFold(%#q, %#q) = %v, want %v", tt.t, tt.s, out, tt.out)
		}
	}
}

func TestBufferGrowNegative(t *testing.T) {
	defer func() {
		if err := recover(); err == nil {
			t.Fatal("Grow(-1) should have panicked")
		}
	}()
	var b Buffer
	b.Grow(-1)
}

func TestBufferTruncateNegative(t *testing.T) {
	defer func() {
		if err := recover(); err == nil {
			t.Fatal("Truncate(-1) should have panicked")
		}
	}()
	var b Buffer
	b.Truncate(-1)
}

func TestBufferTruncateOutOfRange(t *testing.T) {
	defer func() {
		if err := recover(); err == nil {
			t.Fatal("Truncate(20) should have panicked")
		}
	}()
	var b Buffer
	b.Write(make([]byte, 10))
	b.Truncate(20)
}

var containsTests = []struct {
	b, subslice []byte
	want        bool
}{
	{[]byte("hello"), []byte("hel"), true},
	{[]byte("日本語"), []byte("日本"), true},
	{[]byte("hello"), []byte("Hello, world"), false},
	{[]byte("東京"), []byte("京東"), false},
}

func TestContains(t *testing.T) {
	for _, tt := range containsTests {
		if got := Contains(tt.b, tt.subslice); got != tt.want {
			t.Errorf("Contains(%q, %q) = %v, want %v", tt.b, tt.subslice, got, tt.want)
		}
	}
}

var makeFieldsInput = func() []byte {
	x := make([]byte, 1<<20)
	// Input is ~10% space, ~10% 2-byte UTF-8, rest ASCII non-space.
	for i := range x {
		switch rand.Intn(10) {
		case 0:
			x[i] = ' '
		case 1:
			if i > 0 && x[i-1] == 'x' {
				copy(x[i-1:], "χ")
				break
			}
			fallthrough
		default:
			x[i] = 'x'
		}
	}
	return x
}

var fieldsInput = makeFieldsInput()

func BenchmarkFields(b *testing.B) {
	b.SetBytes(int64(len(fieldsInput)))
	for i := 0; i < b.N; i++ {
		Fields(fieldsInput)
	}
}

func BenchmarkFieldsFunc(b *testing.B) {
	b.SetBytes(int64(len(fieldsInput)))
	for i := 0; i < b.N; i++ {
		FieldsFunc(fieldsInput, unicode.IsSpace)
	}
}

func BenchmarkTrimSpace(b *testing.B) {
	s := []byte("  Some text.  \n")
	for i := 0; i < b.N; i++ {
		TrimSpace(s)
	}
}

func BenchmarkRepeat(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Repeat([]byte("-"), 80)
	}
}
