// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jsonv2

package jsonwire

import (
	"cmp"
	"slices"
	"testing"
	"unicode/utf16"
	"unicode/utf8"
)

func TestQuoteRune(t *testing.T) {
	tests := []struct{ in, want string }{
		{"x", `'x'`},
		{"\n", `'\n'`},
		{"'", `'\''`},
		{"\xff", `'\xff'`},
		{"💩", `'💩'`},
		{"💩"[:1], `'\xf0'`},
		{"\uffff", `'\uffff'`},
		{"\U00101234", `'\U00101234'`},
	}
	for _, tt := range tests {
		got := QuoteRune([]byte(tt.in))
		if got != tt.want {
			t.Errorf("quoteRune(%q) = %s, want %s", tt.in, got, tt.want)
		}
	}
}

var compareUTF16Testdata = []string{"", "\r", "1", "f\xfe", "f\xfe\xff", "f\xff", "\u0080", "\u00f6", "\u20ac", "\U0001f600", "\ufb33"}

func TestCompareUTF16(t *testing.T) {
	for i, si := range compareUTF16Testdata {
		for j, sj := range compareUTF16Testdata {
			got := CompareUTF16([]byte(si), []byte(sj))
			want := cmp.Compare(i, j)
			if got != want {
				t.Errorf("CompareUTF16(%q, %q) = %v, want %v", si, sj, got, want)
			}
		}
	}
}

func FuzzCompareUTF16(f *testing.F) {
	for _, td1 := range compareUTF16Testdata {
		for _, td2 := range compareUTF16Testdata {
			f.Add([]byte(td1), []byte(td2))
		}
	}

	// CompareUTF16Simple is identical to CompareUTF16,
	// but relies on naively converting a string to a []uint16 codepoints.
	// It is easy to verify as correct, but is slow.
	CompareUTF16Simple := func(x, y []byte) int {
		ux := utf16.Encode([]rune(string(x)))
		uy := utf16.Encode([]rune(string(y)))
		return slices.Compare(ux, uy)
	}

	f.Fuzz(func(t *testing.T, s1, s2 []byte) {
		// Compare the optimized and simplified implementations.
		got := CompareUTF16(s1, s2)
		want := CompareUTF16Simple(s1, s2)
		if got != want && utf8.Valid(s1) && utf8.Valid(s2) {
			t.Errorf("CompareUTF16(%q, %q) = %v, want %v", s1, s2, got, want)
		}
	})
}

func TestTruncatePointer(t *testing.T) {
	tests := []struct{ in, want string }{
		{"hello", "hello"},
		{"/a/b/c", "/a/b/c"},
		{"/a/b/c/d/e/f/g", "/a/b/…/f/g"},
		{"supercalifragilisticexpialidocious", "super…cious"},
		{"/supercalifragilisticexpialidocious/supercalifragilisticexpialidocious", "/supe…/…cious"},
		{"/supercalifragilisticexpialidocious/supercalifragilisticexpialidocious/supercalifragilisticexpialidocious", "/supe…/…/…cious"},
		{"/a/supercalifragilisticexpialidocious/supercalifragilisticexpialidocious", "/a/…/…cious"},
		{"/supercalifragilisticexpialidocious/supercalifragilisticexpialidocious/b", "/supe…/…/b"},
		{"/fizz/buzz/bazz", "/fizz/…/bazz"},
		{"/fizz/buzz/bazz/razz", "/fizz/…/razz"},
		{"/////////////////////////////", "/////…/////"},
		{"/🎄❤️✨/🎁✅😊/🎅🔥⭐", "/🎄…/…/…⭐"},
	}
	for _, tt := range tests {
		got := TruncatePointer(tt.in, 10)
		if got != tt.want {
			t.Errorf("TruncatePointer(%q) = %q, want %q", tt.in, got, tt.want)
		}
	}

}
