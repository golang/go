// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strings_test

// Derived from bytes/compare_test.go.
// Benchmarks omitted since the underlying implementation is identical.

import (
	. "strings"
	"testing"
)

var compareTests = []struct {
	a, b string
	i    int
}{
	{"", "", 0},
	{"a", "", 1},
	{"", "a", -1},
	{"abc", "abc", 0},
	{"ab", "abc", -1},
	{"abc", "ab", 1},
	{"x", "ab", 1},
	{"ab", "x", -1},
	{"x", "a", 1},
	{"b", "x", -1},
	// test runtime·memeq's chunked implementation
	{"abcdefgh", "abcdefgh", 0},
	{"abcdefghi", "abcdefghi", 0},
	{"abcdefghi", "abcdefghj", -1},
}

func TestCompare(t *testing.T) {
	for _, tt := range compareTests {
		cmp := Compare(tt.a, tt.b)
		if cmp != tt.i {
			t.Errorf(`Compare(%q, %q) = %v`, tt.a, tt.b, cmp)
		}
	}
}

func TestCompareIdenticalString(t *testing.T) {
	var s = "Hello Gophers!"
	if Compare(s, s) != 0 {
		t.Error("s != s")
	}
	if Compare(s, s[:1]) != 1 {
		t.Error("s > s[:1] failed")
	}
}

func TestCompareStrings(t *testing.T) {
	n := 128
	a := make([]byte, n+1)
	b := make([]byte, n+1)
	for len := 0; len < 128; len++ {
		// randomish but deterministic data.  No 0 or 255.
		for i := 0; i < len; i++ {
			a[i] = byte(1 + 31*i%254)
			b[i] = byte(1 + 31*i%254)
		}
		// data past the end is different
		for i := len; i <= n; i++ {
			a[i] = 8
			b[i] = 9
		}

		cmp := Compare(string(a[:len]), string(b[:len]))
		if cmp != 0 {
			t.Errorf(`CompareIdentical(%d) = %d`, len, cmp)
		}
		if len > 0 {
			cmp = Compare(string(a[:len-1]), string(b[:len]))
			if cmp != -1 {
				t.Errorf(`CompareAshorter(%d) = %d`, len, cmp)
			}
			cmp = Compare(string(a[:len]), string(b[:len-1]))
			if cmp != 1 {
				t.Errorf(`CompareBshorter(%d) = %d`, len, cmp)
			}
		}
		for k := 0; k < len; k++ {
			b[k] = a[k] - 1
			cmp = Compare(string(a[:len]), string(b[:len]))
			if cmp != 1 {
				t.Errorf(`CompareAbigger(%d,%d) = %d`, len, k, cmp)
			}
			b[k] = a[k] + 1
			cmp = Compare(string(a[:len]), string(b[:len]))
			if cmp != -1 {
				t.Errorf(`CompareBbigger(%d,%d) = %d`, len, k, cmp)
			}
			b[k] = a[k]
		}
	}
}
