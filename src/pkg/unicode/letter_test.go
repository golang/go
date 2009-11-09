// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unicode_test

import (
	"testing";
	. "unicode";
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
	0x620,
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
	cas, in, out int;
}

var caseTest = []caseT{
	// errors
	caseT{-1, '\n', 0xFFFD},
	caseT{UpperCase, -1, -1},
	caseT{UpperCase, 1<<30, 1<<30},

	// ASCII (special-cased so test carefully)
	caseT{UpperCase, '\n', '\n'},
	caseT{UpperCase, 'a', 'A'},
	caseT{UpperCase, 'A', 'A'},
	caseT{UpperCase, '7', '7'},
	caseT{LowerCase, '\n', '\n'},
	caseT{LowerCase, 'a', 'a'},
	caseT{LowerCase, 'A', 'a'},
	caseT{LowerCase, '7', '7'},
	caseT{TitleCase, '\n', '\n'},
	caseT{TitleCase, 'a', 'A'},
	caseT{TitleCase, 'A', 'A'},
	caseT{TitleCase, '7', '7'},

	// Latin-1: easy to read the tests!
	caseT{UpperCase, 0x80, 0x80},
	caseT{UpperCase, 'Å', 'Å'},
	caseT{UpperCase, 'å', 'Å'},
	caseT{LowerCase, 0x80, 0x80},
	caseT{LowerCase, 'Å', 'å'},
	caseT{LowerCase, 'å', 'å'},
	caseT{TitleCase, 0x80, 0x80},
	caseT{TitleCase, 'Å', 'Å'},
	caseT{TitleCase, 'å', 'Å'},

	// 0131;LATIN SMALL LETTER DOTLESS I;Ll;0;L;;;;;N;;;0049;;0049
	caseT{UpperCase, 0x0131, 'I'},
	caseT{LowerCase, 0x0131, 0x0131},
	caseT{TitleCase, 0x0131, 'I'},

	// 0133;LATIN SMALL LIGATURE IJ;Ll;0;L;<compat> 0069 006A;;;;N;LATIN SMALL LETTER I J;;0132;;0132
	caseT{UpperCase, 0x0133, 0x0132},
	caseT{LowerCase, 0x0133, 0x0133},
	caseT{TitleCase, 0x0133, 0x0132},

	// 212A;KELVIN SIGN;Lu;0;L;004B;;;;N;DEGREES KELVIN;;;006B;
	caseT{UpperCase, 0x212A, 0x212A},
	caseT{LowerCase, 0x212A, 'k'},
	caseT{TitleCase, 0x212A, 0x212A},

	// From an UpperLower sequence
	// A640;CYRILLIC CAPITAL LETTER ZEMLYA;Lu;0;L;;;;;N;;;;A641;
	caseT{UpperCase, 0xA640, 0xA640},
	caseT{LowerCase, 0xA640, 0xA641},
	caseT{TitleCase, 0xA640, 0xA640},
	// A641;CYRILLIC SMALL LETTER ZEMLYA;Ll;0;L;;;;;N;;;A640;;A640
	caseT{UpperCase, 0xA641, 0xA640},
	caseT{LowerCase, 0xA641, 0xA641},
	caseT{TitleCase, 0xA641, 0xA640},
	// A64E;CYRILLIC CAPITAL LETTER NEUTRAL YER;Lu;0;L;;;;;N;;;;A64F;
	caseT{UpperCase, 0xA64E, 0xA64E},
	caseT{LowerCase, 0xA64E, 0xA64F},
	caseT{TitleCase, 0xA64E, 0xA64E},
	// A65F;CYRILLIC SMALL LETTER YN;Ll;0;L;;;;;N;;;A65E;;A65E
	caseT{UpperCase, 0xA65F, 0xA65E},
	caseT{LowerCase, 0xA65F, 0xA65F},
	caseT{TitleCase, 0xA65F, 0xA65E},

	// From another UpperLower sequence
	// 0139;LATIN CAPITAL LETTER L WITH ACUTE;Lu;0;L;004C 0301;;;;N;LATIN CAPITAL LETTER L ACUTE;;;013A;
	caseT{UpperCase, 0x0139, 0x0139},
	caseT{LowerCase, 0x0139, 0x013A},
	caseT{TitleCase, 0x0139, 0x0139},
	// 013F;LATIN CAPITAL LETTER L WITH MIDDLE DOT;Lu;0;L;<compat> 004C 00B7;;;;N;;;;0140;
	caseT{UpperCase, 0x013f, 0x013f},
	caseT{LowerCase, 0x013f, 0x0140},
	caseT{TitleCase, 0x013f, 0x013f},
	// 0148;LATIN SMALL LETTER N WITH CARON;Ll;0;L;006E 030C;;;;N;LATIN SMALL LETTER N HACEK;;0147;;0147
	caseT{UpperCase, 0x0148, 0x0147},
	caseT{LowerCase, 0x0148, 0x0148},
	caseT{TitleCase, 0x0148, 0x0147},

	// Last block in the 5.1.0 table
	// 10400;DESERET CAPITAL LETTER LONG I;Lu;0;L;;;;;N;;;;10428;
	caseT{UpperCase, 0x10400, 0x10400},
	caseT{LowerCase, 0x10400, 0x10428},
	caseT{TitleCase, 0x10400, 0x10400},
	// 10427;DESERET CAPITAL LETTER EW;Lu;0;L;;;;;N;;;;1044F;
	caseT{UpperCase, 0x10427, 0x10427},
	caseT{LowerCase, 0x10427, 0x1044F},
	caseT{TitleCase, 0x10427, 0x10427},
	// 10428;DESERET SMALL LETTER LONG I;Ll;0;L;;;;;N;;;10400;;10400
	caseT{UpperCase, 0x10428, 0x10400},
	caseT{LowerCase, 0x10428, 0x10428},
	caseT{TitleCase, 0x10428, 0x10400},
	// 1044F;DESERET SMALL LETTER EW;Ll;0;L;;;;;N;;;10427;;10427
	caseT{UpperCase, 0x1044F, 0x10427},
	caseT{LowerCase, 0x1044F, 0x1044F},
	caseT{TitleCase, 0x1044F, 0x10427},

	// First one not in the 5.1.0 table
	// 10450;SHAVIAN LETTER PEEP;Lo;0;L;;;;;N;;;;;
	caseT{UpperCase, 0x10450, 0x10450},
	caseT{LowerCase, 0x10450, 0x10450},
	caseT{TitleCase, 0x10450, 0x10450},
}

func TestIsLetter(t *testing.T) {
	for _, r := range upperTest {
		if !IsLetter(r) {
			t.Errorf("IsLetter(U+%04X) = false, want true\n", r)
		}
	}
	for _, r := range letterTest {
		if !IsLetter(r) {
			t.Errorf("IsLetter(U+%04X) = false, want true\n", r)
		}
	}
	for _, r := range notletterTest {
		if IsLetter(r) {
			t.Errorf("IsLetter(U+%04X) = true, want false\n", r)
		}
	}
}

func TestIsUpper(t *testing.T) {
	for _, r := range upperTest {
		if !IsUpper(r) {
			t.Errorf("IsUpper(U+%04X) = false, want true\n", r)
		}
	}
	for _, r := range notupperTest {
		if IsUpper(r) {
			t.Errorf("IsUpper(U+%04X) = true, want false\n", r)
		}
	}
	for _, r := range notletterTest {
		if IsUpper(r) {
			t.Errorf("IsUpper(U+%04X) = true, want false\n", r)
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
	return "ErrorCase";
}

func TestTo(t *testing.T) {
	for _, c := range caseTest {
		r := To(c.cas, c.in);
		if c.out != r {
			t.Errorf("To(U+%04X, %s) = U+%04X want U+%04X\n", c.in, caseString(c.cas), r, c.out)
		}
	}
}

func TestToUpperCase(t *testing.T) {
	for _, c := range caseTest {
		if c.cas != UpperCase {
			continue
		}
		r := ToUpper(c.in);
		if c.out != r {
			t.Errorf("ToUpper(U+%04X) = U+%04X want U+%04X\n", c.in, r, c.out)
		}
	}
}

func TestToLowerCase(t *testing.T) {
	for _, c := range caseTest {
		if c.cas != LowerCase {
			continue
		}
		r := ToLower(c.in);
		if c.out != r {
			t.Errorf("ToLower(U+%04X) = U+%04X want U+%04X\n", c.in, r, c.out)
		}
	}
}

func TestToTitleCase(t *testing.T) {
	for _, c := range caseTest {
		if c.cas != TitleCase {
			continue
		}
		r := ToTitle(c.in);
		if c.out != r {
			t.Errorf("ToTitle(U+%04X) = U+%04X want U+%04X\n", c.in, r, c.out)
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
	for i := 0; i < 0x100; i++ {
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
