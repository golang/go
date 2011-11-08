// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package utf16_test

import (
	"reflect"
	"testing"
	"unicode"
	. "unicode/utf16"
)

type encodeTest struct {
	in  []rune
	out []uint16
}

var encodeTests = []encodeTest{
	{[]rune{1, 2, 3, 4}, []uint16{1, 2, 3, 4}},
	{[]rune{0xffff, 0x10000, 0x10001, 0x12345, 0x10ffff},
		[]uint16{0xffff, 0xd800, 0xdc00, 0xd800, 0xdc01, 0xd808, 0xdf45, 0xdbff, 0xdfff}},
	{[]rune{'a', 'b', 0xd7ff, 0xd800, 0xdfff, 0xe000, 0x110000, -1},
		[]uint16{'a', 'b', 0xd7ff, 0xfffd, 0xfffd, 0xe000, 0xfffd, 0xfffd}},
}

func TestEncode(t *testing.T) {
	for _, tt := range encodeTests {
		out := Encode(tt.in)
		if !reflect.DeepEqual(out, tt.out) {
			t.Errorf("Encode(%x) = %x; want %x", tt.in, out, tt.out)
		}
	}
}

func TestEncodeRune(t *testing.T) {
	for i, tt := range encodeTests {
		j := 0
		for _, r := range tt.in {
			r1, r2 := EncodeRune(r)
			if r < 0x10000 || r > unicode.MaxRune {
				if j >= len(tt.out) {
					t.Errorf("#%d: ran out of tt.out", i)
					break
				}
				if r1 != unicode.ReplacementChar || r2 != unicode.ReplacementChar {
					t.Errorf("EncodeRune(%#x) = %#x, %#x; want 0xfffd, 0xfffd", r, r1, r2)
				}
				j++
			} else {
				if j+1 >= len(tt.out) {
					t.Errorf("#%d: ran out of tt.out", i)
					break
				}
				if r1 != rune(tt.out[j]) || r2 != rune(tt.out[j+1]) {
					t.Errorf("EncodeRune(%#x) = %#x, %#x; want %#x, %#x", r, r1, r2, tt.out[j], tt.out[j+1])
				}
				j += 2
				dec := DecodeRune(r1, r2)
				if dec != r {
					t.Errorf("DecodeRune(%#x, %#x) = %#x; want %#x", r1, r2, dec, r)
				}
			}
		}
		if j != len(tt.out) {
			t.Errorf("#%d: EncodeRune didn't generate enough output", i)
		}
	}
}

type decodeTest struct {
	in  []uint16
	out []rune
}

var decodeTests = []decodeTest{
	{[]uint16{1, 2, 3, 4}, []rune{1, 2, 3, 4}},
	{[]uint16{0xffff, 0xd800, 0xdc00, 0xd800, 0xdc01, 0xd808, 0xdf45, 0xdbff, 0xdfff},
		[]rune{0xffff, 0x10000, 0x10001, 0x12345, 0x10ffff}},
	{[]uint16{0xd800, 'a'}, []rune{0xfffd, 'a'}},
	{[]uint16{0xdfff}, []rune{0xfffd}},
}

func TestDecode(t *testing.T) {
	for _, tt := range decodeTests {
		out := Decode(tt.in)
		if !reflect.DeepEqual(out, tt.out) {
			t.Errorf("Decode(%x) = %x; want %x", tt.in, out, tt.out)
		}
	}
}
