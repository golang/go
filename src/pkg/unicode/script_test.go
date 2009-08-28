// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unicode

import "testing"

type T struct {
	rune	int;
	script	string;
}

// Hand-chosen tests from Unicode 5.1.0, mostly to discover when new
// scripts and categories arise.
var inTest = []T {
	T{0x06e2,	"Arabic"},
	T{0x0567,	"Armenian"},
	T{0x1b37,	"Balinese"},
	T{0x09c2,	"Bengali"},
	T{0x3115,	"Bopomofo"},
	T{0x282d,	"Braille"},
	T{0x1a1a,	"Buginese"},
	T{0x1747,	"Buhid"},
	T{0x156d,	"Canadian_Aboriginal"},
	T{0x102a9,	"Carian"},
	T{0xaa4d,	"Cham"},
	T{0x13c2,	"Cherokee"},
	T{0x0020,	"Common"},
	T{0x1d4a5,	"Common"},
	T{0x2cfc,	"Coptic"},
	T{0x12420,	"Cuneiform"},
	T{0x1080c,	"Cypriot"},
	T{0xa663,	"Cyrillic"},
	T{0x10430,	"Deseret"},
	T{0x094a,	"Devanagari"},
	T{0x1271,	"Ethiopic"},
	T{0x10fc,	"Georgian"},
	T{0x2c40,	"Glagolitic"},
	T{0x10347,	"Gothic"},
	T{0x03ae,	"Greek"},
	T{0x0abf,	"Gujarati"},
	T{0x0a24,	"Gurmukhi"},
	T{0x3028,	"Han"},
	T{0x11b8,	"Hangul"},
	T{0x1727,	"Hanunoo"},
	T{0x05a0,	"Hebrew"},
	T{0x3058,	"Hiragana"},
	T{0x20e6,	"Inherited"},
	T{0x0cbd,	"Kannada"},
	T{0x30a6,	"Katakana"},
	T{0xa928,	"Kayah_Li"},
	T{0x10a11,	"Kharoshthi"},
	T{0x17c6,	"Khmer"},
	T{0x0eaa,	"Lao"},
	T{0x1d79,	"Latin"},
	T{0x1c10,	"Lepcha"},
	T{0x1930,	"Limbu"},
	T{0x1003c,	"Linear_B"},
	T{0x10290,	"Lycian"},
	T{0x10930,	"Lydian"},
	T{0x0d42,	"Malayalam"},
	T{0x1822,	"Mongolian"},
	T{0x104c,	"Myanmar"},
	T{0x19c3,	"New_Tai_Lue"},
	T{0x07f8,	"Nko"},
	T{0x169b,	"Ogham"},
	T{0x1c6a,	"Ol_Chiki"},
	T{0x10310,	"Old_Italic"},
	T{0x103c9,	"Old_Persian"},
	T{0x0b3e,	"Oriya"},
	T{0x10491,	"Osmanya"},
	T{0xa860,	"Phags_Pa"},
	T{0x10918,	"Phoenician"},
	T{0xa949,	"Rejang"},
	T{0x16c0,	"Runic"},
	T{0xa892,	"Saurashtra"},
	T{0x10463,	"Shavian"},
	T{0x0dbd,	"Sinhala"},
	T{0x1ba3,	"Sundanese"},
	T{0xa803,	"Syloti_Nagri"},
	T{0x070f,	"Syriac"},
	T{0x170f,	"Tagalog"},
	T{0x176f,	"Tagbanwa"},
	T{0x1972,	"Tai_Le"},
	T{0x0bbf,	"Tamil"},
	T{0x0c55,	"Telugu"},
	T{0x07a7,	"Thaana"},
	T{0x0e46,	"Thai"},
	T{0x0f36,	"Tibetan"},
	T{0x2d55,	"Tifinagh"},
	T{0x10388,	"Ugaritic"},
	T{0xa60e,	"Vai"},
	T{0xa216,	"Yi"},
}

var outTest = []T {	// not really worth being thorough
	T{0x20,	"Telugu"}
}

var inCategoryTest = []T {
	T{0x0081, "Cc"},
	T{0x17b4, "Cf"},
	T{0xf0000, "Co"},
	T{0xdb80, "Cs"},
	T{0x0236, "Ll"},
	T{0x1d9d, "Lm"},
	T{0x07cf, "Lo"},
	T{0x1f8a, "Lt"},
	T{0x03ff, "Lu"},
	T{0x0bc1, "Mc"},
	T{0x20df, "Me"},
	T{0x07f0, "Mn"},
	T{0x1bb2, "Nd"},
	T{0x10147, "Nl"},
	T{0x2478, "No"},
	T{0xfe33, "Pc"},
	T{0x2011, "Pd"},
	T{0x301e, "Pe"},
	T{0x2e03, "Pf"},
	T{0x2e02, "Pi"},
	T{0x0022, "Po"},
	T{0x2770, "Ps"},
	T{0x00a4, "Sc"},
	T{0xa711, "Sk"},
	T{0x25f9, "Sm"},
	T{0x2108, "So"},
	T{0x2028, "Zl"},
	T{0x2029, "Zp"},
	T{0x202f, "Zs"},
	T{0x04aa, "letter"},
}

func TestScripts(t *testing.T) {
	notTested := make(map[string] bool);
	for k := range Scripts {
		notTested[k] = true
	}
	for i, test := range inTest {
		if !Is(Scripts[test.script], test.rune) {
			t.Errorf("IsScript(%#x, %s) = false, want true\n", test.rune, test.script);
		}
		notTested[test.script] = false, false
	}
	for i, test := range outTest {
		if Is(Scripts[test.script], test.rune) {
			t.Errorf("IsScript(%#x, %s) = true, want false\n", test.rune, test.script);
		}
	}
	for k := range notTested {
		t.Error("not tested:", k)
	}
}

func TestCategories(t *testing.T) {
	notTested := make(map[string] bool);
	for k := range Categories {
		notTested[k] = true
	}
	for i, test := range inCategoryTest {
		if !Is(Categories[test.script], test.rune) {
			t.Errorf("IsCategory(%#x, %s) = false, want true\n", test.rune, test.script);
		}
		notTested[test.script] = false, false
	}
	for k := range notTested {
		t.Error("not tested:", k)
	}
}

