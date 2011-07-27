// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unicode_test

import (
	"testing"
	. "unicode"
)

var upperTest = []int{
	0x41,
	0xc0,
	0xd8,
	0x100,
	0x139,
	0x14a,
	0x178,
	0x181,
	0x376,
	0x3cf,
	0x1f2a,
	0x2102,
	0x2c00,
	0x2c10,
	0x2c20,
	0xa650,
	0xa722,
	0xff3a,
	0x10400,
	0x1d400,
	0x1d7ca,
}

var notupperTest = []int{
	0x40,
	0x5b,
	0x61,
	0x185,
	0x1b0,
	0x377,
	0x387,
	0x2150,
	0xffff,
	0x10000,
}

var letterTest = []int{
	0x41,
	0x61,
	0xaa,
	0xba,
	0xc8,
	0xdb,
	0xf9,
	0x2ec,
	0x535,
	0x620,
	0x6e6,
	0x93d,
	0xa15,
	0xb99,
	0xdc0,
	0xedd,
	0x1000,
	0x1200,
	0x1312,
	0x1401,
	0x1885,
	0x2c00,
	0xa800,
	0xf900,
	0xfa30,
	0xffda,
	0xffdc,
	0x10000,
	0x10300,
	0x10400,
	0x20000,
	0x2f800,
	0x2fa1d,
}

var notletterTest = []int{
	0x20,
	0x35,
	0x375,
	0x619,
	0x700,
	0xfffe,
	0x1ffff,
	0x10ffff,
}

// Contains all the special cased Latin-1 chars.
var spaceTest = []int{
	0x09,
	0x0a,
	0x0b,
	0x0c,
	0x0d,
	0x20,
	0x85,
	0xA0,
	0x2000,
	0x3000,
}

type caseT struct {
	cas, in, out int
}

var caseTest = []caseT{
	// errors
	{-1, '\n', 0xFFFD},
	{UpperCase, -1, -1},
	{UpperCase, 1 << 30, 1 << 30},

	// ASCII (special-cased so test carefully)
	{UpperCase, '\n', '\n'},
	{UpperCase, 'a', 'A'},
	{UpperCase, 'A', 'A'},
	{UpperCase, '7', '7'},
	{LowerCase, '\n', '\n'},
	{LowerCase, 'a', 'a'},
	{LowerCase, 'A', 'a'},
	{LowerCase, '7', '7'},
	{TitleCase, '\n', '\n'},
	{TitleCase, 'a', 'A'},
	{TitleCase, 'A', 'A'},
	{TitleCase, '7', '7'},

	// Latin-1: easy to read the tests!
	{UpperCase, 0x80, 0x80},
	{UpperCase, 'Å', 'Å'},
	{UpperCase, 'å', 'Å'},
	{LowerCase, 0x80, 0x80},
	{LowerCase, 'Å', 'å'},
	{LowerCase, 'å', 'å'},
	{TitleCase, 0x80, 0x80},
	{TitleCase, 'Å', 'Å'},
	{TitleCase, 'å', 'Å'},

	// 0131;LATIN SMALL LETTER DOTLESS I;Ll;0;L;;;;;N;;;0049;;0049
	{UpperCase, 0x0131, 'I'},
	{LowerCase, 0x0131, 0x0131},
	{TitleCase, 0x0131, 'I'},

	// 0133;LATIN SMALL LIGATURE IJ;Ll;0;L;<compat> 0069 006A;;;;N;LATIN SMALL LETTER I J;;0132;;0132
	{UpperCase, 0x0133, 0x0132},
	{LowerCase, 0x0133, 0x0133},
	{TitleCase, 0x0133, 0x0132},

	// 212A;KELVIN SIGN;Lu;0;L;004B;;;;N;DEGREES KELVIN;;;006B;
	{UpperCase, 0x212A, 0x212A},
	{LowerCase, 0x212A, 'k'},
	{TitleCase, 0x212A, 0x212A},

	// From an UpperLower sequence
	// A640;CYRILLIC CAPITAL LETTER ZEMLYA;Lu;0;L;;;;;N;;;;A641;
	{UpperCase, 0xA640, 0xA640},
	{LowerCase, 0xA640, 0xA641},
	{TitleCase, 0xA640, 0xA640},
	// A641;CYRILLIC SMALL LETTER ZEMLYA;Ll;0;L;;;;;N;;;A640;;A640
	{UpperCase, 0xA641, 0xA640},
	{LowerCase, 0xA641, 0xA641},
	{TitleCase, 0xA641, 0xA640},
	// A64E;CYRILLIC CAPITAL LETTER NEUTRAL YER;Lu;0;L;;;;;N;;;;A64F;
	{UpperCase, 0xA64E, 0xA64E},
	{LowerCase, 0xA64E, 0xA64F},
	{TitleCase, 0xA64E, 0xA64E},
	// A65F;CYRILLIC SMALL LETTER YN;Ll;0;L;;;;;N;;;A65E;;A65E
	{UpperCase, 0xA65F, 0xA65E},
	{LowerCase, 0xA65F, 0xA65F},
	{TitleCase, 0xA65F, 0xA65E},

	// From another UpperLower sequence
	// 0139;LATIN CAPITAL LETTER L WITH ACUTE;Lu;0;L;004C 0301;;;;N;LATIN CAPITAL LETTER L ACUTE;;;013A;
	{UpperCase, 0x0139, 0x0139},
	{LowerCase, 0x0139, 0x013A},
	{TitleCase, 0x0139, 0x0139},
	// 013F;LATIN CAPITAL LETTER L WITH MIDDLE DOT;Lu;0;L;<compat> 004C 00B7;;;;N;;;;0140;
	{UpperCase, 0x013f, 0x013f},
	{LowerCase, 0x013f, 0x0140},
	{TitleCase, 0x013f, 0x013f},
	// 0148;LATIN SMALL LETTER N WITH CARON;Ll;0;L;006E 030C;;;;N;LATIN SMALL LETTER N HACEK;;0147;;0147
	{UpperCase, 0x0148, 0x0147},
	{LowerCase, 0x0148, 0x0148},
	{TitleCase, 0x0148, 0x0147},

	// Last block in the 5.1.0 table
	// 10400;DESERET CAPITAL LETTER LONG I;Lu;0;L;;;;;N;;;;10428;
	{UpperCase, 0x10400, 0x10400},
	{LowerCase, 0x10400, 0x10428},
	{TitleCase, 0x10400, 0x10400},
	// 10427;DESERET CAPITAL LETTER EW;Lu;0;L;;;;;N;;;;1044F;
	{UpperCase, 0x10427, 0x10427},
	{LowerCase, 0x10427, 0x1044F},
	{TitleCase, 0x10427, 0x10427},
	// 10428;DESERET SMALL LETTER LONG I;Ll;0;L;;;;;N;;;10400;;10400
	{UpperCase, 0x10428, 0x10400},
	{LowerCase, 0x10428, 0x10428},
	{TitleCase, 0x10428, 0x10400},
	// 1044F;DESERET SMALL LETTER EW;Ll;0;L;;;;;N;;;10427;;10427
	{UpperCase, 0x1044F, 0x10427},
	{LowerCase, 0x1044F, 0x1044F},
	{TitleCase, 0x1044F, 0x10427},

	// First one not in the 5.1.0 table
	// 10450;SHAVIAN LETTER PEEP;Lo;0;L;;;;;N;;;;;
	{UpperCase, 0x10450, 0x10450},
	{LowerCase, 0x10450, 0x10450},
	{TitleCase, 0x10450, 0x10450},

	// Non-letters with case.
	{LowerCase, 0x2161, 0x2171},
	{UpperCase, 0x0345, 0x0399},
}

func TestIsLetter(t *testing.T) {
	for _, r := range upperTest {
		if !IsLetter(r) {
			t.Errorf("IsLetter(U+%04X) = false, want true", r)
		}
	}
	for _, r := range letterTest {
		if !IsLetter(r) {
			t.Errorf("IsLetter(U+%04X) = false, want true", r)
		}
	}
	for _, r := range notletterTest {
		if IsLetter(r) {
			t.Errorf("IsLetter(U+%04X) = true, want false", r)
		}
	}
}

func TestIsUpper(t *testing.T) {
	for _, r := range upperTest {
		if !IsUpper(r) {
			t.Errorf("IsUpper(U+%04X) = false, want true", r)
		}
	}
	for _, r := range notupperTest {
		if IsUpper(r) {
			t.Errorf("IsUpper(U+%04X) = true, want false", r)
		}
	}
	for _, r := range notletterTest {
		if IsUpper(r) {
			t.Errorf("IsUpper(U+%04X) = true, want false", r)
		}
	}
}

func caseString(c int) string {
	switch c {
	case UpperCase:
		return "UpperCase"
	case LowerCase:
		return "LowerCase"
	case TitleCase:
		return "TitleCase"
	}
	return "ErrorCase"
}

func TestTo(t *testing.T) {
	for _, c := range caseTest {
		r := To(c.cas, c.in)
		if c.out != r {
			t.Errorf("To(U+%04X, %s) = U+%04X want U+%04X", c.in, caseString(c.cas), r, c.out)
		}
	}
}

func TestToUpperCase(t *testing.T) {
	for _, c := range caseTest {
		if c.cas != UpperCase {
			continue
		}
		r := ToUpper(c.in)
		if c.out != r {
			t.Errorf("ToUpper(U+%04X) = U+%04X want U+%04X", c.in, r, c.out)
		}
	}
}

func TestToLowerCase(t *testing.T) {
	for _, c := range caseTest {
		if c.cas != LowerCase {
			continue
		}
		r := ToLower(c.in)
		if c.out != r {
			t.Errorf("ToLower(U+%04X) = U+%04X want U+%04X", c.in, r, c.out)
		}
	}
}

func TestToTitleCase(t *testing.T) {
	for _, c := range caseTest {
		if c.cas != TitleCase {
			continue
		}
		r := ToTitle(c.in)
		if c.out != r {
			t.Errorf("ToTitle(U+%04X) = U+%04X want U+%04X", c.in, r, c.out)
		}
	}
}

func TestIsSpace(t *testing.T) {
	for _, c := range spaceTest {
		if !IsSpace(c) {
			t.Errorf("IsSpace(U+%04X) = false; want true", c)
		}
	}
	for _, c := range letterTest {
		if IsSpace(c) {
			t.Errorf("IsSpace(U+%04X) = true; want false", c)
		}
	}
}

// Check that the optimizations for IsLetter etc. agree with the tables.
// We only need to check the Latin-1 range.
func TestLetterOptimizations(t *testing.T) {
	for i := 0; i <= MaxLatin1; i++ {
		if Is(Letter, i) != IsLetter(i) {
			t.Errorf("IsLetter(U+%04X) disagrees with Is(Letter)", i)
		}
		if Is(Upper, i) != IsUpper(i) {
			t.Errorf("IsUpper(U+%04X) disagrees with Is(Upper)", i)
		}
		if Is(Lower, i) != IsLower(i) {
			t.Errorf("IsLower(U+%04X) disagrees with Is(Lower)", i)
		}
		if Is(Title, i) != IsTitle(i) {
			t.Errorf("IsTitle(U+%04X) disagrees with Is(Title)", i)
		}
		if Is(White_Space, i) != IsSpace(i) {
			t.Errorf("IsSpace(U+%04X) disagrees with Is(White_Space)", i)
		}
		if To(UpperCase, i) != ToUpper(i) {
			t.Errorf("ToUpper(U+%04X) disagrees with To(Upper)", i)
		}
		if To(LowerCase, i) != ToLower(i) {
			t.Errorf("ToLower(U+%04X) disagrees with To(Lower)", i)
		}
		if To(TitleCase, i) != ToTitle(i) {
			t.Errorf("ToTitle(U+%04X) disagrees with To(Title)", i)
		}
	}
}

func TestTurkishCase(t *testing.T) {
	lower := []int("abcçdefgğhıijklmnoöprsştuüvyz")
	upper := []int("ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ")
	for i, l := range lower {
		u := upper[i]
		if TurkishCase.ToLower(l) != l {
			t.Errorf("lower(U+%04X) is U+%04X not U+%04X", l, TurkishCase.ToLower(l), l)
		}
		if TurkishCase.ToUpper(u) != u {
			t.Errorf("upper(U+%04X) is U+%04X not U+%04X", u, TurkishCase.ToUpper(u), u)
		}
		if TurkishCase.ToUpper(l) != u {
			t.Errorf("upper(U+%04X) is U+%04X not U+%04X", l, TurkishCase.ToUpper(l), u)
		}
		if TurkishCase.ToLower(u) != l {
			t.Errorf("lower(U+%04X) is U+%04X not U+%04X", u, TurkishCase.ToLower(l), l)
		}
		if TurkishCase.ToTitle(u) != u {
			t.Errorf("title(U+%04X) is U+%04X not U+%04X", u, TurkishCase.ToTitle(u), u)
		}
		if TurkishCase.ToTitle(l) != u {
			t.Errorf("title(U+%04X) is U+%04X not U+%04X", l, TurkishCase.ToTitle(l), u)
		}
	}
}

var simpleFoldTests = []string{
	// SimpleFold could order its returned slices in any order it wants,
	// but we know it orders them in increasing order starting at in
	// and looping around from MaxRune to 0.

	// Easy cases.
	"Aa",
	"aA",
	"δΔ",
	"Δδ",

	// ASCII special cases.
	"KkK",
	"kKK",
	"KKk",
	"Ssſ",
	"sſS",
	"ſSs",

	// Non-ASCII special cases.
	"ρϱΡ",
	"ϱΡρ",
	"Ρρϱ",
	"ͅΙιι",
	"Ιιιͅ",
	"ιιͅΙ",
	"ιͅΙι",

	// Extra special cases: has lower/upper but no case fold.
	"İ",
	"ı",
}

func TestSimpleFold(t *testing.T) {
	for _, tt := range simpleFoldTests {
		cycle := []int(tt)
		rune := cycle[len(cycle)-1]
		for _, out := range cycle {
			if r := SimpleFold(rune); r != out {
				t.Errorf("SimpleFold(%#U) = %#U, want %#U", rune, r, out)
			}
			rune = out
		}
	}
}
