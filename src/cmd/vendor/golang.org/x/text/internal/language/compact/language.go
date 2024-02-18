// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:generate go run gen.go gen_index.go -output tables.go
//go:generate go run gen_parents.go

package compact

// TODO: Remove above NOTE after:
// - verifying that tables are dropped correctly (most notably matcher tables).

import (
	"strings"

	"golang.org/x/text/internal/language"
)

// Tag represents a BCP 47 language tag. It is used to specify an instance of a
// specific language or locale. All language tag values are guaranteed to be
// well-formed.
type Tag struct {
	// NOTE: exported tags will become part of the public API.
	language ID
	locale   ID
	full     fullTag // always a language.Tag for now.
}

const _und = 0

type fullTag interface {
	IsRoot() bool
	Parent() language.Tag
}

// Make a compact Tag from a fully specified internal language Tag.
func Make(t language.Tag) (tag Tag) {
	if region := t.TypeForKey("rg"); len(region) == 6 && region[2:] == "zzzz" {
		if r, err := language.ParseRegion(region[:2]); err == nil {
			tFull := t
			t, _ = t.SetTypeForKey("rg", "")
			// TODO: should we not consider "va" for the language tag?
			var exact1, exact2 bool
			tag.language, exact1 = FromTag(t)
			t.RegionID = r
			tag.locale, exact2 = FromTag(t)
			if !exact1 || !exact2 {
				tag.full = tFull
			}
			return tag
		}
	}
	lang, ok := FromTag(t)
	tag.language = lang
	tag.locale = lang
	if !ok {
		tag.full = t
	}
	return tag
}

// Tag returns an internal language Tag version of this tag.
func (t Tag) Tag() language.Tag {
	if t.full != nil {
		return t.full.(language.Tag)
	}
	tag := t.language.Tag()
	if t.language != t.locale {
		loc := t.locale.Tag()
		tag, _ = tag.SetTypeForKey("rg", strings.ToLower(loc.RegionID.String())+"zzzz")
	}
	return tag
}

// IsCompact reports whether this tag is fully defined in terms of ID.
func (t *Tag) IsCompact() bool {
	return t.full == nil
}

// MayHaveVariants reports whether a tag may have variants. If it returns false
// it is guaranteed the tag does not have variants.
func (t Tag) MayHaveVariants() bool {
	return t.full != nil || int(t.language) >= len(coreTags)
}

// MayHaveExtensions reports whether a tag may have extensions. If it returns
// false it is guaranteed the tag does not have them.
func (t Tag) MayHaveExtensions() bool {
	return t.full != nil ||
		int(t.language) >= len(coreTags) ||
		t.language != t.locale
}

// IsRoot returns true if t is equal to language "und".
func (t Tag) IsRoot() bool {
	if t.full != nil {
		return t.full.IsRoot()
	}
	return t.language == _und
}

// Parent returns the CLDR parent of t. In CLDR, missing fields in data for a
// specific language are substituted with fields from the parent language.
// The parent for a language may change for newer versions of CLDR.
func (t Tag) Parent() Tag {
	if t.full != nil {
		return Make(t.full.Parent())
	}
	if t.language != t.locale {
		// Simulate stripping -u-rg-xxxxxx
		return Tag{language: t.language, locale: t.language}
	}
	// TODO: use parent lookup table once cycle from internal package is
	// removed. Probably by internalizing the table and declaring this fast
	// enough.
	// lang := compactID(internal.Parent(uint16(t.language)))
	lang, _ := FromTag(t.language.Tag().Parent())
	return Tag{language: lang, locale: lang}
}

// nextToken returns token t and the rest of the string.
func nextToken(s string) (t, tail string) {
	p := strings.Index(s[1:], "-")
	if p == -1 {
		return s[1:], ""
	}
	p++
	return s[1:p], s[p:]
}

// LanguageID returns an index, where 0 <= index < NumCompactTags, for tags
// for which data exists in the text repository.The index will change over time
// and should not be stored in persistent storage. If t does not match a compact
// index, exact will be false and the compact index will be returned for the
// first match after repeatedly taking the Parent of t.
func LanguageID(t Tag) (id ID, exact bool) {
	return t.language, t.full == nil
}

// RegionalID returns the ID for the regional variant of this tag. This index is
// used to indicate region-specific overrides, such as default currency, default
// calendar and week data, default time cycle, and default measurement system
// and unit preferences.
//
// For instance, the tag en-GB-u-rg-uszzzz specifies British English with US
// settings for currency, number formatting, etc. The CompactIndex for this tag
// will be that for en-GB, while the RegionalID will be the one corresponding to
// en-US.
func RegionalID(t Tag) (id ID, exact bool) {
	return t.locale, t.full == nil
}

// LanguageTag returns t stripped of regional variant indicators.
//
// At the moment this means it is stripped of a regional and variant subtag "rg"
// and "va" in the "u" extension.
func (t Tag) LanguageTag() Tag {
	if t.full == nil {
		return Tag{language: t.language, locale: t.language}
	}
	tt := t.Tag()
	tt.SetTypeForKey("rg", "")
	tt.SetTypeForKey("va", "")
	return Make(tt)
}

// RegionalTag returns the regional variant of the tag.
//
// At the moment this means that the region is set from the regional subtag
// "rg" in the "u" extension.
func (t Tag) RegionalTag() Tag {
	rt := Tag{language: t.locale, locale: t.locale}
	if t.full == nil {
		return rt
	}
	b := language.Builder{}
	tag := t.Tag()
	// tag, _ = tag.SetTypeForKey("rg", "")
	b.SetTag(t.locale.Tag())
	if v := tag.Variants(); v != "" {
		for _, v := range strings.Split(v, "-") {
			b.AddVariant(v)
		}
	}
	for _, e := range tag.Extensions() {
		b.AddExt(e)
	}
	return t
}

// FromTag reports closest matching ID for an internal language Tag.
func FromTag(t language.Tag) (id ID, exact bool) {
	// TODO: perhaps give more frequent tags a lower index.
	// TODO: we could make the indexes stable. This will excluded some
	//       possibilities for optimization, so don't do this quite yet.
	exact = true

	b, s, r := t.Raw()
	if t.HasString() {
		if t.IsPrivateUse() {
			// We have no entries for user-defined tags.
			return 0, false
		}
		hasExtra := false
		if t.HasVariants() {
			if t.HasExtensions() {
				build := language.Builder{}
				build.SetTag(language.Tag{LangID: b, ScriptID: s, RegionID: r})
				build.AddVariant(t.Variants())
				exact = false
				t = build.Make()
			}
			hasExtra = true
		} else if _, ok := t.Extension('u'); ok {
			// TODO: va may mean something else. Consider not considering it.
			// Strip all but the 'va' entry.
			old := t
			variant := t.TypeForKey("va")
			t = language.Tag{LangID: b, ScriptID: s, RegionID: r}
			if variant != "" {
				t, _ = t.SetTypeForKey("va", variant)
				hasExtra = true
			}
			exact = old == t
		} else {
			exact = false
		}
		if hasExtra {
			// We have some variants.
			for i, s := range specialTags {
				if s == t {
					return ID(i + len(coreTags)), exact
				}
			}
			exact = false
		}
	}
	if x, ok := getCoreIndex(t); ok {
		return x, exact
	}
	exact = false
	if r != 0 && s == 0 {
		// Deal with cases where an extra script is inserted for the region.
		t, _ := t.Maximize()
		if x, ok := getCoreIndex(t); ok {
			return x, exact
		}
	}
	for t = t.Parent(); t != root; t = t.Parent() {
		// No variants specified: just compare core components.
		// The key has the form lllssrrr, where l, s, and r are nibbles for
		// respectively the langID, scriptID, and regionID.
		if x, ok := getCoreIndex(t); ok {
			return x, exact
		}
	}
	return 0, exact
}

var root = language.Tag{}
