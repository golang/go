// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package json

import (
	"bytes"
	"strings"
	"testing"
	"unicode/utf8"
)

var foldTests = []struct {
	fn   func(s, t []byte) bool
	s, t string
	want bool
}{
	{equalFoldRight, "", "", true},
	{equalFoldRight, "a", "a", true},
	{equalFoldRight, "", "a", false},
	{equalFoldRight, "a", "", false},
	{equalFoldRight, "a", "A", true},
	{equalFoldRight, "AB", "ab", true},
	{equalFoldRight, "AB", "ac", false},
	{equalFoldRight, "sbkKc", "ſbKKc", true},
	{equalFoldRight, "SbKkc", "ſbKKc", true},
	{equalFoldRight, "SbKkc", "ſbKK", false},
	{equalFoldRight, "e", "é", false},
	{equalFoldRight, "s", "S", true},

	{simpleLetterEqualFold, "", "", true},
	{simpleLetterEqualFold, "abc", "abc", true},
	{simpleLetterEqualFold, "abc", "ABC", true},
	{simpleLetterEqualFold, "abc", "ABCD", false},
	{simpleLetterEqualFold, "abc", "xxx", false},

	{asciiEqualFold, "a_B", "A_b", true},
	{asciiEqualFold, "aa@", "aa`", false}, // verify 0x40 and 0x60 aren't case-equivalent
}

func TestFold(t *testing.T) {
	for i, tt := range foldTests {
		if got := tt.fn([]byte(tt.s), []byte(tt.t)); got != tt.want {
			t.Errorf("%d. %q, %q = %v; want %v", i, tt.s, tt.t, got, tt.want)
		}
		truth := strings.EqualFold(tt.s, tt.t)
		if truth != tt.want {
			t.Errorf("strings.EqualFold doesn't agree with case %d", i)
		}
	}
}

func TestFoldAgainstUnicode(t *testing.T) {
	const bufSize = 5
	buf1 := make([]byte, 0, bufSize)
	buf2 := make([]byte, 0, bufSize)
	var runes []rune
	for i := 0x20; i <= 0x7f; i++ {
		runes = append(runes, rune(i))
	}
	runes = append(runes, kelvin, smallLongEss)

	funcs := []struct {
		name   string
		fold   func(s, t []byte) bool
		letter bool // must be ASCII letter
		simple bool // must be simple ASCII letter (not 'S' or 'K')
	}{
		{
			name: "equalFoldRight",
			fold: equalFoldRight,
		},
		{
			name:   "asciiEqualFold",
			fold:   asciiEqualFold,
			simple: true,
		},
		{
			name:   "simpleLetterEqualFold",
			fold:   simpleLetterEqualFold,
			simple: true,
			letter: true,
		},
	}

	for _, ff := range funcs {
		for _, r := range runes {
			if r >= utf8.RuneSelf {
				continue
			}
			if ff.letter && !isASCIILetter(byte(r)) {
				continue
			}
			if ff.simple && (r == 's' || r == 'S' || r == 'k' || r == 'K') {
				continue
			}
			for _, r2 := range runes {
				buf1 := append(buf1[:0], 'x')
				buf2 := append(buf2[:0], 'x')
				buf1 = buf1[:1+utf8.EncodeRune(buf1[1:bufSize], r)]
				buf2 = buf2[:1+utf8.EncodeRune(buf2[1:bufSize], r2)]
				buf1 = append(buf1, 'x')
				buf2 = append(buf2, 'x')
				want := bytes.EqualFold(buf1, buf2)
				if got := ff.fold(buf1, buf2); got != want {
					t.Errorf("%s(%q, %q) = %v; want %v", ff.name, buf1, buf2, got, want)
				}
			}
		}
	}
}

func isASCIILetter(b byte) bool {
	return ('A' <= b && b <= 'Z') || ('a' <= b && b <= 'z')
}
