// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"runtime"
	"strconv"
	"strings"
	"testing"
	"unicode/utf8"
)

// Strings and slices that don't escape and fit into tmpBuf are stack allocated,
// which defeats using AllocsPerRun to test other optimizations.
const sizeNoStack = 100

func BenchmarkCompareStringEqual(b *testing.B) {
	bytes := []byte("Hello Gophers!")
	s1, s2 := string(bytes), string(bytes)
	for i := 0; i < b.N; i++ {
		if s1 != s2 {
			b.Fatal("s1 != s2")
		}
	}
}

func BenchmarkCompareStringIdentical(b *testing.B) {
	s1 := "Hello Gophers!"
	s2 := s1
	for i := 0; i < b.N; i++ {
		if s1 != s2 {
			b.Fatal("s1 != s2")
		}
	}
}

func BenchmarkCompareStringSameLength(b *testing.B) {
	s1 := "Hello Gophers!"
	s2 := "Hello, Gophers"
	for i := 0; i < b.N; i++ {
		if s1 == s2 {
			b.Fatal("s1 == s2")
		}
	}
}

func BenchmarkCompareStringDifferentLength(b *testing.B) {
	s1 := "Hello Gophers!"
	s2 := "Hello, Gophers!"
	for i := 0; i < b.N; i++ {
		if s1 == s2 {
			b.Fatal("s1 == s2")
		}
	}
}

func BenchmarkCompareStringBigUnaligned(b *testing.B) {
	bytes := make([]byte, 0, 1<<20)
	for len(bytes) < 1<<20 {
		bytes = append(bytes, "Hello Gophers!"...)
	}
	s1, s2 := string(bytes), "hello"+string(bytes)
	for i := 0; i < b.N; i++ {
		if s1 != s2[len("hello"):] {
			b.Fatal("s1 != s2")
		}
	}
	b.SetBytes(int64(len(s1)))
}

func BenchmarkCompareStringBig(b *testing.B) {
	bytes := make([]byte, 0, 1<<20)
	for len(bytes) < 1<<20 {
		bytes = append(bytes, "Hello Gophers!"...)
	}
	s1, s2 := string(bytes), string(bytes)
	for i := 0; i < b.N; i++ {
		if s1 != s2 {
			b.Fatal("s1 != s2")
		}
	}
	b.SetBytes(int64(len(s1)))
}

func BenchmarkConcatStringAndBytes(b *testing.B) {
	s1 := []byte("Gophers!")
	for i := 0; i < b.N; i++ {
		_ = "Hello " + string(s1)
	}
}

var escapeString string

func BenchmarkSliceByteToString(b *testing.B) {
	buf := []byte{'!'}
	for n := 0; n < 8; n++ {
		b.Run(strconv.Itoa(len(buf)), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				escapeString = string(buf)
			}
		})
		buf = append(buf, buf...)
	}
}

var stringdata = []struct{ name, data string }{
	{"ASCII", "01234567890"},
	{"Japanese", "日本語日本語日本語"},
	{"MixedLength", "$Ѐࠀက퀀𐀀\U00040000\U0010FFFF"},
}

var sinkInt int

func BenchmarkRuneCount(b *testing.B) {
	// Each sub-benchmark counts the runes in a string in a different way.
	b.Run("lenruneslice", func(b *testing.B) {
		for _, sd := range stringdata {
			b.Run(sd.name, func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					sinkInt += len([]rune(sd.data))
				}
			})
		}
	})
	b.Run("rangeloop", func(b *testing.B) {
		for _, sd := range stringdata {
			b.Run(sd.name, func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					n := 0
					for range sd.data {
						n++
					}
					sinkInt += n
				}
			})
		}
	})
	b.Run("utf8.RuneCountInString", func(b *testing.B) {
		for _, sd := range stringdata {
			b.Run(sd.name, func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					sinkInt += utf8.RuneCountInString(sd.data)
				}
			})
		}
	})
}

func BenchmarkRuneIterate(b *testing.B) {
	b.Run("range", func(b *testing.B) {
		for _, sd := range stringdata {
			b.Run(sd.name, func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					for range sd.data {
					}
				}
			})
		}
	})
	b.Run("range1", func(b *testing.B) {
		for _, sd := range stringdata {
			b.Run(sd.name, func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					for range sd.data {
					}
				}
			})
		}
	})
	b.Run("range2", func(b *testing.B) {
		for _, sd := range stringdata {
			b.Run(sd.name, func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					for range sd.data {
					}
				}
			})
		}
	})
}

func BenchmarkArrayEqual(b *testing.B) {
	a1 := [16]byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	a2 := [16]byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if a1 != a2 {
			b.Fatal("not equal")
		}
	}
}

func TestStringW(t *testing.T) {
	strings := []string{
		"hello",
		"a\u5566\u7788b",
	}

	for _, s := range strings {
		var b []uint16
		for _, c := range s {
			b = append(b, uint16(c))
			if c != rune(uint16(c)) {
				t.Errorf("bad test: stringW can't handle >16 bit runes")
			}
		}
		b = append(b, 0)
		r := runtime.GostringW(b)
		if r != s {
			t.Errorf("gostringW(%v) = %s, want %s", b, r, s)
		}
	}
}

func TestLargeStringConcat(t *testing.T) {
	output := runTestProg(t, "testprog", "stringconcat")
	want := "panic: " + strings.Repeat("0", 1<<10) + strings.Repeat("1", 1<<10) +
		strings.Repeat("2", 1<<10) + strings.Repeat("3", 1<<10)
	if !strings.HasPrefix(output, want) {
		t.Fatalf("output does not start with %q:\n%s", want, output)
	}
}

func TestCompareTempString(t *testing.T) {
	s := strings.Repeat("x", sizeNoStack)
	b := []byte(s)
	n := testing.AllocsPerRun(1000, func() {
		if string(b) != s {
			t.Fatalf("strings are not equal: '%v' and '%v'", string(b), s)
		}
		if string(b) == s {
		} else {
			t.Fatalf("strings are not equal: '%v' and '%v'", string(b), s)
		}
	})
	if n != 0 {
		t.Fatalf("want 0 allocs, got %v", n)
	}
}

func TestStringIndexHaystack(t *testing.T) {
	// See issue 25864.
	haystack := []byte("hello")
	needle := "ll"
	n := testing.AllocsPerRun(1000, func() {
		if strings.Index(string(haystack), needle) != 2 {
			t.Fatalf("needle not found")
		}
	})
	if n != 0 {
		t.Fatalf("want 0 allocs, got %v", n)
	}
}

func TestStringIndexNeedle(t *testing.T) {
	// See issue 25864.
	haystack := "hello"
	needle := []byte("ll")
	n := testing.AllocsPerRun(1000, func() {
		if strings.Index(haystack, string(needle)) != 2 {
			t.Fatalf("needle not found")
		}
	})
	if n != 0 {
		t.Fatalf("want 0 allocs, got %v", n)
	}
}

func TestStringOnStack(t *testing.T) {
	s := ""
	for i := 0; i < 3; i++ {
		s = "a" + s + "b" + s + "c"
	}

	if want := "aaabcbabccbaabcbabccc"; s != want {
		t.Fatalf("want: '%v', got '%v'", want, s)
	}
}

func TestIntString(t *testing.T) {
	// Non-escaping result of intstring.
	s := ""
	for i := rune(0); i < 4; i++ {
		s += string(i+'0') + string(i+'0'+1)
	}
	if want := "01122334"; s != want {
		t.Fatalf("want '%v', got '%v'", want, s)
	}

	// Escaping result of intstring.
	var a [4]string
	for i := rune(0); i < 4; i++ {
		a[i] = string(i + '0')
	}
	s = a[0] + a[1] + a[2] + a[3]
	if want := "0123"; s != want {
		t.Fatalf("want '%v', got '%v'", want, s)
	}
}

func TestIntStringAllocs(t *testing.T) {
	unknown := '0'
	n := testing.AllocsPerRun(1000, func() {
		s1 := string(unknown)
		s2 := string(unknown + 1)
		if s1 == s2 {
			t.Fatalf("bad")
		}
	})
	if n != 0 {
		t.Fatalf("want 0 allocs, got %v", n)
	}
}

func TestRangeStringCast(t *testing.T) {
	s := strings.Repeat("x", sizeNoStack)
	n := testing.AllocsPerRun(1000, func() {
		for i, c := range []byte(s) {
			if c != s[i] {
				t.Fatalf("want '%c' at pos %v, got '%c'", s[i], i, c)
			}
		}
	})
	if n != 0 {
		t.Fatalf("want 0 allocs, got %v", n)
	}
}

func isZeroed(b []byte) bool {
	for _, x := range b {
		if x != 0 {
			return false
		}
	}
	return true
}

func isZeroedR(r []rune) bool {
	for _, x := range r {
		if x != 0 {
			return false
		}
	}
	return true
}

func TestString2Slice(t *testing.T) {
	// Make sure we don't return slices that expose
	// an unzeroed section of stack-allocated temp buf
	// between len and cap. See issue 14232.
	s := "foož"
	b := ([]byte)(s)
	if !isZeroed(b[len(b):cap(b)]) {
		t.Errorf("extra bytes not zeroed")
	}
	r := ([]rune)(s)
	if !isZeroedR(r[len(r):cap(r)]) {
		t.Errorf("extra runes not zeroed")
	}
}

const intSize = 32 << (^uint(0) >> 63)

type atoi64Test struct {
	in  string
	out int64
	ok  bool
}

var atoi64tests = []atoi64Test{
	{"", 0, false},
	{"0", 0, true},
	{"-0", 0, true},
	{"1", 1, true},
	{"-1", -1, true},
	{"12345", 12345, true},
	{"-12345", -12345, true},
	{"012345", 12345, true},
	{"-012345", -12345, true},
	{"12345x", 0, false},
	{"-12345x", 0, false},
	{"98765432100", 98765432100, true},
	{"-98765432100", -98765432100, true},
	{"20496382327982653440", 0, false},
	{"-20496382327982653440", 0, false},
	{"9223372036854775807", 1<<63 - 1, true},
	{"-9223372036854775807", -(1<<63 - 1), true},
	{"9223372036854775808", 0, false},
	{"-9223372036854775808", -1 << 63, true},
	{"9223372036854775809", 0, false},
	{"-9223372036854775809", 0, false},
}

func TestAtoi(t *testing.T) {
	switch intSize {
	case 32:
		for i := range atoi32tests {
			test := &atoi32tests[i]
			out, ok := runtime.Atoi(test.in)
			if test.out != int32(out) || test.ok != ok {
				t.Errorf("atoi(%q) = (%v, %v) want (%v, %v)",
					test.in, out, ok, test.out, test.ok)
			}
		}
	case 64:
		for i := range atoi64tests {
			test := &atoi64tests[i]
			out, ok := runtime.Atoi(test.in)
			if test.out != int64(out) || test.ok != ok {
				t.Errorf("atoi(%q) = (%v, %v) want (%v, %v)",
					test.in, out, ok, test.out, test.ok)
			}
		}
	}
}

type atoi32Test struct {
	in  string
	out int32
	ok  bool
}

var atoi32tests = []atoi32Test{
	{"", 0, false},
	{"0", 0, true},
	{"-0", 0, true},
	{"1", 1, true},
	{"-1", -1, true},
	{"12345", 12345, true},
	{"-12345", -12345, true},
	{"012345", 12345, true},
	{"-012345", -12345, true},
	{"12345x", 0, false},
	{"-12345x", 0, false},
	{"987654321", 987654321, true},
	{"-987654321", -987654321, true},
	{"2147483647", 1<<31 - 1, true},
	{"-2147483647", -(1<<31 - 1), true},
	{"2147483648", 0, false},
	{"-2147483648", -1 << 31, true},
	{"2147483649", 0, false},
	{"-2147483649", 0, false},
}

func TestAtoi32(t *testing.T) {
	for i := range atoi32tests {
		test := &atoi32tests[i]
		out, ok := runtime.Atoi32(test.in)
		if test.out != out || test.ok != ok {
			t.Errorf("atoi32(%q) = (%v, %v) want (%v, %v)",
				test.in, out, ok, test.out, test.ok)
		}
	}
}

type parseReleaseTest struct {
	in                  string
	major, minor, patch int
}

var parseReleaseTests = []parseReleaseTest{
	{"", -1, -1, -1},
	{"x", -1, -1, -1},
	{"5", 5, 0, 0},
	{"5.12", 5, 12, 0},
	{"5.12-x", 5, 12, 0},
	{"5.12.1", 5, 12, 1},
	{"5.12.1-x", 5, 12, 1},
	{"5.12.1.0", 5, 12, 1},
	{"5.20496382327982653440", -1, -1, -1},
}

func TestParseRelease(t *testing.T) {
	for _, test := range parseReleaseTests {
		major, minor, patch, ok := runtime.ParseRelease(test.in)
		if !ok {
			major, minor, patch = -1, -1, -1
		}
		if test.major != major || test.minor != minor || test.patch != patch {
			t.Errorf("parseRelease(%q) = (%v, %v, %v) want (%v, %v, %v)",
				test.in, major, minor, patch,
				test.major, test.minor, test.patch)
		}
	}
}
