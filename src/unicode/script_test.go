// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unicode_test

import (
	"testing"
	. "unicode"
)

type T struct {
	rune   rune
	script string
}

var inCategoryTest = []T{
	{0x0081, "Cc"},
	{0x200B, "Cf"},
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
	// Unifieds.
	{0x04aa, "L"},
	{0x0009, "C"},
	{0x1712, "M"},
	{0x0031, "N"},
	{0x00bb, "P"},
	{0x00a2, "S"},
	{0x00a0, "Z"},
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
	{0x2065, "Other_Default_Ignorable_Code_Point"},
	{0x0BD7, "Other_Grapheme_Extend"},
	{0x0387, "Other_ID_Continue"},
	{0x212E, "Other_ID_Start"},
	{0x2094, "Other_Lowercase"},
	{0x2040, "Other_Math"},
	{0x216F, "Other_Uppercase"},
	{0x0027, "Pattern_Syntax"},
	{0x0020, "Pattern_White_Space"},
	{0x06DD, "Prepended_Concatenation_Mark"},
	{0x300D, "Quotation_Mark"},
	{0x2EF3, "Radical"},
	{0x1f1ff, "Regional_Indicator"},
	{0x061F, "STerm"}, // Deprecated alias of Sentence_Terminal
	{0x061F, "Sentence_Terminal"},
	{0x2071, "Soft_Dotted"},
	{0x003A, "Terminal_Punctuation"},
	{0x9FC3, "Unified_Ideograph"},
	{0xFE0F, "Variation_Selector"},
	{0x0020, "White_Space"},
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
			t.Errorf("IsCategory(%U, %s) = false, want true", test.rune, test.script)
		}
		delete(notTested, test.script)
	}
	for k := range notTested {
		t.Error("category not tested:", k)
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
			t.Errorf("IsCategory(%U, %s) = false, want true", test.rune, test.script)
		}
		delete(notTested, test.script)
	}
	for k := range notTested {
		t.Error("property not tested:", k)
	}
}
