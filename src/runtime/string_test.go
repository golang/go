// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"runtime"
	"strings"
	"testing"
)

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
	output := executeTest(t, largeStringConcatSource, nil)
	want := "panic: " + strings.Repeat("0", 1<<10) + strings.Repeat("1", 1<<10) +
		strings.Repeat("2", 1<<10) + strings.Repeat("3", 1<<10)
	if !strings.HasPrefix(output, want) {
		t.Fatalf("output does not start with %q:\n%s", want, output)
	}
}

var largeStringConcatSource = `
package main
import "strings"
func main() {
	s0 := strings.Repeat("0", 1<<10)
	s1 := strings.Repeat("1", 1<<10)
	s2 := strings.Repeat("2", 1<<10)
	s3 := strings.Repeat("3", 1<<10)
	s := s0 + s1 + s2 + s3
	panic(s)
}
`

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
	s := "foo"
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
