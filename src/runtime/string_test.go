// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"runtime"
	"strings"
	"testing"
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

func BenchmarkRuneIterate(b *testing.B) {
	bytes := make([]byte, 100)
	for i := range bytes {
		bytes[i] = byte('A')
	}
	s := string(bytes)
	for i := 0; i < b.N; i++ {
		for range s {
		}
	}
}

func BenchmarkRuneIterate2(b *testing.B) {
	bytes := make([]byte, 100)
	for i := range bytes {
		bytes[i] = byte('A')
	}
	s := string(bytes)
	for i := 0; i < b.N; i++ {
		for range s {
		}
	}
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

func TestGostringnocopy(t *testing.T) {
	max := *runtime.Maxstring
	b := make([]byte, max+10)
	for i := uintptr(0); i < max+9; i++ {
		b[i] = 'a'
	}
	_ = runtime.Gostringnocopy(&b[0])
	newmax := *runtime.Maxstring
	if newmax != max+9 {
		t.Errorf("want %d, got %d", max+9, newmax)
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
	for i := 0; i < 4; i++ {
		s += string(i+'0') + string(i+'0'+1)
	}
	if want := "01122334"; s != want {
		t.Fatalf("want '%v', got '%v'", want, s)
	}

	// Escaping result of intstring.
	var a [4]string
	for i := 0; i < 4; i++ {
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
