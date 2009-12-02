// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unicode_test

import (
	"testing";
	. "unicode";
)

type T struct {
	rune	int;
	script	string;
}

// Hand-chosen tests from Unicode 5.1.0, mostly to discover when new
// scripts and categories arise.
var inTest = []T{
	T{0x06e2, "Arabic"},
	T{0x0567, "Armenian"},
	T{0x10b20, "Avestan"},
	T{0x1b37, "Balinese"},
	T{0xa6af, "Bamum"},
	T{0x09c2, "Bengali"},
	T{0x3115, "Bopomofo"},
	T{0x282d, "Braille"},
	T{0x1a1a, "Buginese"},
	T{0x1747, "Buhid"},
	T{0x156d, "Canadian_Aboriginal"},
	T{0x102a9, "Carian"},
	T{0xaa4d, "Cham"},
	T{0x13c2, "Cherokee"},
	T{0x0020, "Common"},
	T{0x1d4a5, "Common"},
	T{0x2cfc, "Coptic"},
	T{0x12420, "Cuneiform"},
	T{0x1080c, "Cypriot"},
	T{0xa663, "Cyrillic"},
	T{0x10430, "Deseret"},
	T{0x094a, "Devanagari"},
	T{0x13001, "Egyptian_Hieroglyphs"},
	T{0x1271, "Ethiopic"},
	T{0x10fc, "Georgian"},
	T{0x2c40, "Glagolitic"},
	T{0x10347, "Gothic"},
	T{0x03ae, "Greek"},
	T{0x0abf, "Gujarati"},
	T{0x0a24, "Gurmukhi"},
	T{0x3028, "Han"},
	T{0x11b8, "Hangul"},
	T{0x1727, "Hanunoo"},
	T{0x05a0, "Hebrew"},
	T{0x3058, "Hiragana"},
	T{0x10841, "Imperial_Aramaic"},
	T{0x20e6, "Inherited"},
	T{0x10b70, "Inscriptional_Pahlavi"},
	T{0x10b5a, "Inscriptional_Parthian"},
	T{0xa9d0, "Javanese"},
	T{0x1109f, "Kaithi"},
	T{0x0cbd, "Kannada"},
	T{0x30a6, "Katakana"},
	T{0xa928, "Kayah_Li"},
	T{0x10a11, "Kharoshthi"},
	T{0x17c6, "Khmer"},
	T{0x0eaa, "Lao"},
	T{0x1d79, "Latin"},
	T{0x1c10, "Lepcha"},
	T{0x1930, "Limbu"},
	T{0x1003c, "Linear_B"},
	T{0xa4e1, "Lisu"},
	T{0x10290, "Lycian"},
	T{0x10930, "Lydian"},
	T{0x0d42, "Malayalam"},
	T{0xabd0, "Meetei_Mayek"},
	T{0x1822, "Mongolian"},
	T{0x104c, "Myanmar"},
	T{0x19c3, "New_Tai_Lue"},
	T{0x07f8, "Nko"},
	T{0x169b, "Ogham"},
	T{0x1c6a, "Ol_Chiki"},
	T{0x10310, "Old_Italic"},
	T{0x103c9, "Old_Persian"},
	T{0x10a6f, "Old_South_Arabian"},
	T{0x10c20, "Old_Turkic"},
	T{0x0b3e, "Oriya"},
	T{0x10491, "Osmanya"},
	T{0xa860, "Phags_Pa"},
	T{0x10918, "Phoenician"},
	T{0xa949, "Rejang"},
	T{0x16c0, "Runic"},
	T{0x081d, "Samaritan"},
	T{0xa892, "Saurashtra"},
	T{0x10463, "Shavian"},
	T{0x0dbd, "Sinhala"},
	T{0x1ba3, "Sundanese"},
	T{0xa803, "Syloti_Nagri"},
	T{0x070f, "Syriac"},
	T{0x170f, "Tagalog"},
	T{0x176f, "Tagbanwa"},
	T{0x1972, "Tai_Le"},
	T{0x1a62, "Tai_Tham"},
	T{0xaadc, "Tai_Viet"},
	T{0x0bbf, "Tamil"},
	T{0x0c55, "Telugu"},
	T{0x07a7, "Thaana"},
	T{0x0e46, "Thai"},
	T{0x0f36, "Tibetan"},
	T{0x2d55, "Tifinagh"},
	T{0x10388, "Ugaritic"},
	T{0xa60e, "Vai"},
	T{0xa216, "Yi"},
}

var outTest = []T{	// not really worth being thorough
	T{0x20, "Telugu"},
}

var inCategoryTest = []T{
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

var inPropTest = []T{
	T{0x0046, "ASCII_Hex_Digit"},
	T{0x200F, "Bidi_Control"},
	T{0x2212, "Dash"},
	T{0xE0001, "Deprecated"},
	T{0x00B7, "Diacritic"},
	T{0x30FE, "Extender"},
	T{0xFF46, "Hex_Digit"},
	T{0x2E17, "Hyphen"},
	T{0x2FFB, "IDS_Binary_Operator"},
	T{0x2FF3, "IDS_Trinary_Operator"},
	T{0xFA6A, "Ideographic"},
	T{0x200D, "Join_Control"},
	T{0x0EC4, "Logical_Order_Exception"},
	T{0x2FFFF, "Noncharacter_Code_Point"},
	T{0x065E, "Other_Alphabetic"},
	T{0x2069, "Other_Default_Ignorable_Code_Point"},
	T{0x0BD7, "Other_Grapheme_Extend"},
	T{0x0387, "Other_ID_Continue"},
	T{0x212E, "Other_ID_Start"},
	T{0x2094, "Other_Lowercase"},
	T{0x2040, "Other_Math"},
	T{0x216F, "Other_Uppercase"},
	T{0x0027, "Pattern_Syntax"},
	T{0x0020, "Pattern_White_Space"},
	T{0x300D, "Quotation_Mark"},
	T{0x2EF3, "Radical"},
	T{0x061F, "STerm"},
	T{0x2071, "Soft_Dotted"},
	T{0x003A, "Terminal_Punctuation"},
	T{0x9FC3, "Unified_Ideograph"},
	T{0xFE0F, "Variation_Selector"},
	T{0x0020, "White_Space"},
}

func TestScripts(t *testing.T) {
	notTested := make(map[string]bool);
	for k := range Scripts {
		notTested[k] = true
	}
	for _, test := range inTest {
		if _, ok := Scripts[test.script]; !ok {
			t.Fatal(test.script, "not a known script")
		}
		if !Is(Scripts[test.script], test.rune) {
			t.Errorf("IsScript(%#x, %s) = false, want true\n", test.rune, test.script)
		}
		notTested[test.script] = false, false;
	}
	for _, test := range outTest {
		if Is(Scripts[test.script], test.rune) {
			t.Errorf("IsScript(%#x, %s) = true, want false\n", test.rune, test.script)
		}
	}
	for k := range notTested {
		t.Error("not tested:", k)
	}
}

func TestCategories(t *testing.T) {
	notTested := make(map[string]bool);
	for k := range Categories {
		notTested[k] = true
	}
	for _, test := range inCategoryTest {
		if _, ok := Categories[test.script]; !ok {
			t.Fatal(test.script, "not a known category")
		}
		if !Is(Categories[test.script], test.rune) {
			t.Errorf("IsCategory(%#x, %s) = false, want true\n", test.rune, test.script)
		}
		notTested[test.script] = false, false;
	}
	for k := range notTested {
		t.Error("not tested:", k)
	}
}

func TestProperties(t *testing.T) {
	notTested := make(map[string]bool);
	for k := range Properties {
		notTested[k] = true
	}
	for _, test := range inPropTest {
		if _, ok := Properties[test.script]; !ok {
			t.Fatal(test.script, "not a known prop")
		}
		if !Is(Properties[test.script], test.rune) {
			t.Errorf("IsCategory(%#x, %s) = false, want true\n", test.rune, test.script)
		}
		notTested[test.script] = false, false;
	}
	for k := range notTested {
		t.Error("not tested:", k)
	}
}
