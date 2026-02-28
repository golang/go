// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package language

import "golang.org/x/text/internal/language/compact"

// TODO: Various sets of commonly use tags and regions.

// MustParse is like Parse, but panics if the given BCP 47 tag cannot be parsed.
// It simplifies safe initialization of Tag values.
func MustParse(s string) Tag {
	t, err := Parse(s)
	if err != nil {
		panic(err)
	}
	return t
}

// MustParse is like Parse, but panics if the given BCP 47 tag cannot be parsed.
// It simplifies safe initialization of Tag values.
func (c CanonType) MustParse(s string) Tag {
	t, err := c.Parse(s)
	if err != nil {
		panic(err)
	}
	return t
}

// MustParseBase is like ParseBase, but panics if the given base cannot be parsed.
// It simplifies safe initialization of Base values.
func MustParseBase(s string) Base {
	b, err := ParseBase(s)
	if err != nil {
		panic(err)
	}
	return b
}

// MustParseScript is like ParseScript, but panics if the given script cannot be
// parsed. It simplifies safe initialization of Script values.
func MustParseScript(s string) Script {
	scr, err := ParseScript(s)
	if err != nil {
		panic(err)
	}
	return scr
}

// MustParseRegion is like ParseRegion, but panics if the given region cannot be
// parsed. It simplifies safe initialization of Region values.
func MustParseRegion(s string) Region {
	r, err := ParseRegion(s)
	if err != nil {
		panic(err)
	}
	return r
}

var (
	und = Tag{}

	Und Tag = Tag{}

	Afrikaans            Tag = Tag(compact.Afrikaans)
	Amharic              Tag = Tag(compact.Amharic)
	Arabic               Tag = Tag(compact.Arabic)
	ModernStandardArabic Tag = Tag(compact.ModernStandardArabic)
	Azerbaijani          Tag = Tag(compact.Azerbaijani)
	Bulgarian            Tag = Tag(compact.Bulgarian)
	Bengali              Tag = Tag(compact.Bengali)
	Catalan              Tag = Tag(compact.Catalan)
	Czech                Tag = Tag(compact.Czech)
	Danish               Tag = Tag(compact.Danish)
	German               Tag = Tag(compact.German)
	Greek                Tag = Tag(compact.Greek)
	English              Tag = Tag(compact.English)
	AmericanEnglish      Tag = Tag(compact.AmericanEnglish)
	BritishEnglish       Tag = Tag(compact.BritishEnglish)
	Spanish              Tag = Tag(compact.Spanish)
	EuropeanSpanish      Tag = Tag(compact.EuropeanSpanish)
	LatinAmericanSpanish Tag = Tag(compact.LatinAmericanSpanish)
	Estonian             Tag = Tag(compact.Estonian)
	Persian              Tag = Tag(compact.Persian)
	Finnish              Tag = Tag(compact.Finnish)
	Filipino             Tag = Tag(compact.Filipino)
	French               Tag = Tag(compact.French)
	CanadianFrench       Tag = Tag(compact.CanadianFrench)
	Gujarati             Tag = Tag(compact.Gujarati)
	Hebrew               Tag = Tag(compact.Hebrew)
	Hindi                Tag = Tag(compact.Hindi)
	Croatian             Tag = Tag(compact.Croatian)
	Hungarian            Tag = Tag(compact.Hungarian)
	Armenian             Tag = Tag(compact.Armenian)
	Indonesian           Tag = Tag(compact.Indonesian)
	Icelandic            Tag = Tag(compact.Icelandic)
	Italian              Tag = Tag(compact.Italian)
	Japanese             Tag = Tag(compact.Japanese)
	Georgian             Tag = Tag(compact.Georgian)
	Kazakh               Tag = Tag(compact.Kazakh)
	Khmer                Tag = Tag(compact.Khmer)
	Kannada              Tag = Tag(compact.Kannada)
	Korean               Tag = Tag(compact.Korean)
	Kirghiz              Tag = Tag(compact.Kirghiz)
	Lao                  Tag = Tag(compact.Lao)
	Lithuanian           Tag = Tag(compact.Lithuanian)
	Latvian              Tag = Tag(compact.Latvian)
	Macedonian           Tag = Tag(compact.Macedonian)
	Malayalam            Tag = Tag(compact.Malayalam)
	Mongolian            Tag = Tag(compact.Mongolian)
	Marathi              Tag = Tag(compact.Marathi)
	Malay                Tag = Tag(compact.Malay)
	Burmese              Tag = Tag(compact.Burmese)
	Nepali               Tag = Tag(compact.Nepali)
	Dutch                Tag = Tag(compact.Dutch)
	Norwegian            Tag = Tag(compact.Norwegian)
	Punjabi              Tag = Tag(compact.Punjabi)
	Polish               Tag = Tag(compact.Polish)
	Portuguese           Tag = Tag(compact.Portuguese)
	BrazilianPortuguese  Tag = Tag(compact.BrazilianPortuguese)
	EuropeanPortuguese   Tag = Tag(compact.EuropeanPortuguese)
	Romanian             Tag = Tag(compact.Romanian)
	Russian              Tag = Tag(compact.Russian)
	Sinhala              Tag = Tag(compact.Sinhala)
	Slovak               Tag = Tag(compact.Slovak)
	Slovenian            Tag = Tag(compact.Slovenian)
	Albanian             Tag = Tag(compact.Albanian)
	Serbian              Tag = Tag(compact.Serbian)
	SerbianLatin         Tag = Tag(compact.SerbianLatin)
	Swedish              Tag = Tag(compact.Swedish)
	Swahili              Tag = Tag(compact.Swahili)
	Tamil                Tag = Tag(compact.Tamil)
	Telugu               Tag = Tag(compact.Telugu)
	Thai                 Tag = Tag(compact.Thai)
	Turkish              Tag = Tag(compact.Turkish)
	Ukrainian            Tag = Tag(compact.Ukrainian)
	Urdu                 Tag = Tag(compact.Urdu)
	Uzbek                Tag = Tag(compact.Uzbek)
	Vietnamese           Tag = Tag(compact.Vietnamese)
	Chinese              Tag = Tag(compact.Chinese)
	SimplifiedChinese    Tag = Tag(compact.SimplifiedChinese)
	TraditionalChinese   Tag = Tag(compact.TraditionalChinese)
	Zulu                 Tag = Tag(compact.Zulu)
)
