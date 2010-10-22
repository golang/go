// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unicode_test

import (
	"testing"
	. "unicode"
)

type T struct {
	rune   int
	script string
}

// Hand-chosen tests from Unicode 5.1.0, mostly to discover when new
// scripts and categories arise.
var inTest = []T{
	{0x06e2, "Arabic"},
	{0x0567, "Armenian"},
	{0x10b20, "Avestan"},
	{0x1b37, "Balinese"},
	{0xa6af, "Bamum"},
	{0x09c2, "Bengali"},
	{0x3115, "Bopomofo"},
	{0x282d, "Braille"},
	{0x1a1a, "Buginese"},
	{0x1747, "Buhid"},
	{0x156d, "Canadian_Aboriginal"},
	{0x102a9, "Carian"},
	{0xaa4d, "Cham"},
	{0x13c2, "Cherokee"},
	{0x0020, "Common"},
	{0x1d4a5, "Common"},
	{0x2cfc, "Coptic"},
	{0x12420, "Cuneiform"},
	{0x1080c, "Cypriot"},
	{0xa663, "Cyrillic"},
	{0x10430, "Deseret"},
	{0x094a, "Devanagari"},
	{0x13001, "Egyptian_Hieroglyphs"},
	{0x1271, "Ethiopic"},
	{0x10fc, "Georgian"},
	{0x2c40, "Glagolitic"},
	{0x10347, "Gothic"},
	{0x03ae, "Greek"},
	{0x0abf, "Gujarati"},
	{0x0a24, "Gurmukhi"},
	{0x3028, "Han"},
	{0x11b8, "Hangul"},
	{0x1727, "Hanunoo"},
	{0x05a0, "Hebrew"},
	{0x3058, "Hiragana"},
	{0x10841, "Imperial_Aramaic"},
	{0x20e6, "Inherited"},
	{0x10b70, "Inscriptional_Pahlavi"},
	{0x10b5a, "Inscriptional_Parthian"},
	{0xa9d0, "Javanese"},
	{0x1109f, "Kaithi"},
	{0x0cbd, "Kannada"},
	{0x30a6, "Katakana"},
	{0xa928, "Kayah_Li"},
	{0x10a11, "Kharoshthi"},
	{0x17c6, "Khmer"},
	{0x0eaa, "Lao"},
	{0x1d79, "Latin"},
	{0x1c10, "Lepcha"},
	{0x1930, "Limbu"},
	{0x1003c, "Linear_B"},
	{0xa4e1, "Lisu"},
	{0x10290, "Lycian"},
	{0x10930, "Lydian"},
	{0x0d42, "Malayalam"},
	{0xabd0, "Meetei_Mayek"},
	{0x1822, "Mongolian"},
	{0x104c, "Myanmar"},
	{0x19c3, "New_Tai_Lue"},
	{0x07f8, "Nko"},
	{0x169b, "Ogham"},
	{0x1c6a, "Ol_Chiki"},
	{0x10310, "Old_Italic"},
	{0x103c9, "Old_Persian"},
	{0x10a6f, "Old_South_Arabian"},
	{0x10c20, "Old_Turkic"},
	{0x0b3e, "Oriya"},
	{0x10491, "Osmanya"},
	{0xa860, "Phags_Pa"},
	{0x10918, "Phoenician"},
	{0xa949, "Rejang"},
	{0x16c0, "Runic"},
	{0x081d, "Samaritan"},
	{0xa892, "Saurashtra"},
	{0x10463, "Shavian"},
	{0x0dbd, "Sinhala"},
	{0x1ba3, "Sundanese"},
	{0xa803, "Syloti_Nagri"},
	{0x070f, "Syriac"},
	{0x170f, "Tagalog"},
	{0x176f, "Tagbanwa"},
	{0x1972, "Tai_Le"},
	{0x1a62, "Tai_Tham"},
	{0xaadc, "Tai_Viet"},
	{0x0bbf, "Tamil"},
	{0x0c55, "Telugu"},
	{0x07a7, "Thaana"},
	{0x0e46, "Thai"},
	{0x0f36, "Tibetan"},
	{0x2d55, "Tifinagh"},
	{0x10388, "Ugaritic"},
	{0xa60e, "Vai"},
	{0xa216, "Yi"},
}

var outTest = []T{ // not really worth being thorough
	{0x20, "Telugu"},
}

var inCategoryTest = []T{
	{0x0081, "Cc"},
	{0x17b4, "Cf"},
	{0xf0000, "Co"},
	{0xdb80, "Cs"},
	{0x0236, "Ll"},
	{0x1d9d, "Lm"},
	{0x07cf, "Lo"},
	{0x1f8a, "Lt"},
	{0x03ff, "Lu"},
	{0x0bc1, "Mc"},
	{0x20df, "Me"},
	{0x07f0, "Mn"},
	{0x1bb2, "Nd"},
	{0x10147, "Nl"},
	{0x2478, "No"},
	{0xfe33, "Pc"},
	{0x2011, "Pd"},
	{0x301e, "Pe"},
	{0x2e03, "Pf"},
	{0x2e02, "Pi"},
	{0x0022, "Po"},
	{0x2770, "Ps"},
	{0x00a4, "Sc"},
	{0xa711, "Sk"},
	{0x25f9, "Sm"},
	{0x2108, "So"},
	{0x2028, "Zl"},
	{0x2029, "Zp"},
	{0x202f, "Zs"},
	{0x04aa, "letter"},
}

var inPropTest = []T{
	{0x0046, "ASCII_Hex_Digit"},
	{0x200F, "Bidi_Control"},
	{0x2212, "Dash"},
	{0xE0001, "Deprecated"},
	{0x00B7, "Diacritic"},
	{0x30FE, "Extender"},
	{0xFF46, "Hex_Digit"},
	{0x2E17, "Hyphen"},
	{0x2FFB, "IDS_Binary_Operator"},
	{0x2FF3, "IDS_Trinary_Operator"},
	{0xFA6A, "Ideographic"},
	{0x200D, "Join_Control"},
	{0x0EC4, "Logical_Order_Exception"},
	{0x2FFFF, "Noncharacter_Code_Point"},
	{0x065E, "Other_Alphabetic"},
	{0x2069, "Other_Default_Ignorable_Code_Point"},
	{0x0BD7, "Other_Grapheme_Extend"},
	{0x0387, "Other_ID_Continue"},
	{0x212E, "Other_ID_Start"},
	{0x2094, "Other_Lowercase"},
	{0x2040, "Other_Math"},
	{0x216F, "Other_Uppercase"},
	{0x0027, "Pattern_Syntax"},
	{0x0020, "Pattern_White_Space"},
	{0x300D, "Quotation_Mark"},
	{0x2EF3, "Radical"},
	{0x061F, "STerm"},
	{0x2071, "Soft_Dotted"},
	{0x003A, "Terminal_Punctuation"},
	{0x9FC3, "Unified_Ideograph"},
	{0xFE0F, "Variation_Selector"},
	{0x0020, "White_Space"},
}

func TestScripts(t *testing.T) {
	notTested := make(map[string]bool)
	for k := range Scripts {
		notTested[k] = true
	}
	for _, test := range inTest {
		if _, ok := Scripts[test.script]; !ok {
			t.Fatal(test.script, "not a known script")
		}
		if !Is(Scripts[test.script], test.rune) {
			t.Errorf("IsScript(%#x, %s) = false, want true", test.rune, test.script)
		}
		notTested[test.script] = false, false
	}
	for _, test := range outTest {
		if Is(Scripts[test.script], test.rune) {
			t.Errorf("IsScript(%#x, %s) = true, want false", test.rune, test.script)
		}
	}
	for k := range notTested {
		t.Error("not tested:", k)
	}
}

func TestCategories(t *testing.T) {
	notTested := make(map[string]bool)
	for k := range Categories {
		notTested[k] = true
	}
	for _, test := range inCategoryTest {
		if _, ok := Categories[test.script]; !ok {
			t.Fatal(test.script, "not a known category")
		}
		if !Is(Categories[test.script], test.rune) {
			t.Errorf("IsCategory(%#x, %s) = false, want true", test.rune, test.script)
		}
		notTested[test.script] = false, false
	}
	for k := range notTested {
		t.Error("not tested:", k)
	}
}

func TestProperties(t *testing.T) {
	notTested := make(map[string]bool)
	for k := range Properties {
		notTested[k] = true
	}
	for _, test := range inPropTest {
		if _, ok := Properties[test.script]; !ok {
			t.Fatal(test.script, "not a known prop")
		}
		if !Is(Properties[test.script], test.rune) {
			t.Errorf("IsCategory(%#x, %s) = false, want true", test.rune, test.script)
		}
		notTested[test.script] = false, false
	}
	for k := range notTested {
		t.Error("not tested:", k)
	}
}
