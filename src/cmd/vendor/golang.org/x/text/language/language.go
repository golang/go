// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:generate go run gen.go -output tables.go

package language

// TODO: Remove above NOTE after:
// - verifying that tables are dropped correctly (most notably matcher tables).

import (
	"strings"

	"golang.org/x/text/internal/language"
	"golang.org/x/text/internal/language/compact"
)

// Tag represents a BCP 47 language tag. It is used to specify an instance of a
// specific language or locale. All language tag values are guaranteed to be
// well-formed.
type Tag compact.Tag

func makeTag(t language.Tag) (tag Tag) {
	return Tag(compact.Make(t))
}

func (t *Tag) tag() language.Tag {
	return (*compact.Tag)(t).Tag()
}

func (t *Tag) isCompact() bool {
	return (*compact.Tag)(t).IsCompact()
}

// TODO: improve performance.
func (t *Tag) lang() language.Language { return t.tag().LangID }
func (t *Tag) region() language.Region { return t.tag().RegionID }
func (t *Tag) script() language.Script { return t.tag().ScriptID }

// Make is a convenience wrapper for Parse that omits the error.
// In case of an error, a sensible default is returned.
func Make(s string) Tag {
	return Default.Make(s)
}

// Make is a convenience wrapper for c.Parse that omits the error.
// In case of an error, a sensible default is returned.
func (c CanonType) Make(s string) Tag {
	t, _ := c.Parse(s)
	return t
}

// Raw returns the raw base language, script and region, without making an
// attempt to infer their values.
func (t Tag) Raw() (b Base, s Script, r Region) {
	tt := t.tag()
	return Base{tt.LangID}, Script{tt.ScriptID}, Region{tt.RegionID}
}

// IsRoot returns true if t is equal to language "und".
func (t Tag) IsRoot() bool {
	return compact.Tag(t).IsRoot()
}

// CanonType can be used to enable or disable various types of canonicalization.
type CanonType int

const (
	// Replace deprecated base languages with their preferred replacements.
	DeprecatedBase CanonType = 1 << iota
	// Replace deprecated scripts with their preferred replacements.
	DeprecatedScript
	// Replace deprecated regions with their preferred replacements.
	DeprecatedRegion
	// Remove redundant scripts.
	SuppressScript
	// Normalize legacy encodings. This includes legacy languages defined in
	// CLDR as well as bibliographic codes defined in ISO-639.
	Legacy
	// Map the dominant language of a macro language group to the macro language
	// subtag. For example cmn -> zh.
	Macro
	// The CLDR flag should be used if full compatibility with CLDR is required.
	// There are a few cases where language.Tag may differ from CLDR. To follow all
	// of CLDR's suggestions, use All|CLDR.
	CLDR

	// Raw can be used to Compose or Parse without Canonicalization.
	Raw CanonType = 0

	// Replace all deprecated tags with their preferred replacements.
	Deprecated = DeprecatedBase | DeprecatedScript | DeprecatedRegion

	// All canonicalizations recommended by BCP 47.
	BCP47 = Deprecated | SuppressScript

	// All canonicalizations.
	All = BCP47 | Legacy | Macro

	// Default is the canonicalization used by Parse, Make and Compose. To
	// preserve as much information as possible, canonicalizations that remove
	// potentially valuable information are not included. The Matcher is
	// designed to recognize similar tags that would be the same if
	// they were canonicalized using All.
	Default = Deprecated | Legacy

	canonLang = DeprecatedBase | Legacy | Macro

	// TODO: LikelyScript, LikelyRegion: suppress similar to ICU.
)

// canonicalize returns the canonicalized equivalent of the tag and
// whether there was any change.
func canonicalize(c CanonType, t language.Tag) (language.Tag, bool) {
	if c == Raw {
		return t, false
	}
	changed := false
	if c&SuppressScript != 0 {
		if t.LangID.SuppressScript() == t.ScriptID {
			t.ScriptID = 0
			changed = true
		}
	}
	if c&canonLang != 0 {
		for {
			if l, aliasType := t.LangID.Canonicalize(); l != t.LangID {
				switch aliasType {
				case language.Legacy:
					if c&Legacy != 0 {
						if t.LangID == _sh && t.ScriptID == 0 {
							t.ScriptID = _Latn
						}
						t.LangID = l
						changed = true
					}
				case language.Macro:
					if c&Macro != 0 {
						// We deviate here from CLDR. The mapping "nb" -> "no"
						// qualifies as a typical Macro language mapping.  However,
						// for legacy reasons, CLDR maps "no", the macro language
						// code for Norwegian, to the dominant variant "nb". This
						// change is currently under consideration for CLDR as well.
						// See https://unicode.org/cldr/trac/ticket/2698 and also
						// https://unicode.org/cldr/trac/ticket/1790 for some of the
						// practical implications. TODO: this check could be removed
						// if CLDR adopts this change.
						if c&CLDR == 0 || t.LangID != _nb {
							changed = true
							t.LangID = l
						}
					}
				case language.Deprecated:
					if c&DeprecatedBase != 0 {
						if t.LangID == _mo && t.RegionID == 0 {
							t.RegionID = _MD
						}
						t.LangID = l
						changed = true
						// Other canonicalization types may still apply.
						continue
					}
				}
			} else if c&Legacy != 0 && t.LangID == _no && c&CLDR != 0 {
				t.LangID = _nb
				changed = true
			}
			break
		}
	}
	if c&DeprecatedScript != 0 {
		if t.ScriptID == _Qaai {
			changed = true
			t.ScriptID = _Zinh
		}
	}
	if c&DeprecatedRegion != 0 {
		if r := t.RegionID.Canonicalize(); r != t.RegionID {
			changed = true
			t.RegionID = r
		}
	}
	return t, changed
}

// Canonicalize returns the canonicalized equivalent of the tag.
func (c CanonType) Canonicalize(t Tag) (Tag, error) {
	// First try fast path.
	if t.isCompact() {
		if _, changed := canonicalize(c, compact.Tag(t).Tag()); !changed {
			return t, nil
		}
	}
	// It is unlikely that one will canonicalize a tag after matching. So do
	// a slow but simple approach here.
	if tag, changed := canonicalize(c, t.tag()); changed {
		tag.RemakeString()
		return makeTag(tag), nil
	}
	return t, nil

}

// Confidence indicates the level of certainty for a given return value.
// For example, Serbian may be written in Cyrillic or Latin script.
// The confidence level indicates whether a value was explicitly specified,
// whether it is typically the only possible value, or whether there is
// an ambiguity.
type Confidence int

const (
	No    Confidence = iota // full confidence that there was no match
	Low                     // most likely value picked out of a set of alternatives
	High                    // value is generally assumed to be the correct match
	Exact                   // exact match or explicitly specified value
)

var confName = []string{"No", "Low", "High", "Exact"}

func (c Confidence) String() string {
	return confName[c]
}

// String returns the canonical string representation of the language tag.
func (t Tag) String() string {
	return t.tag().String()
}

// MarshalText implements encoding.TextMarshaler.
func (t Tag) MarshalText() (text []byte, err error) {
	return t.tag().MarshalText()
}

// UnmarshalText implements encoding.TextUnmarshaler.
func (t *Tag) UnmarshalText(text []byte) error {
	var tag language.Tag
	err := tag.UnmarshalText(text)
	*t = makeTag(tag)
	return err
}

// Base returns the base language of the language tag. If the base language is
// unspecified, an attempt will be made to infer it from the context.
// It uses a variant of CLDR's Add Likely Subtags algorithm. This is subject to change.
func (t Tag) Base() (Base, Confidence) {
	if b := t.lang(); b != 0 {
		return Base{b}, Exact
	}
	tt := t.tag()
	c := High
	if tt.ScriptID == 0 && !tt.RegionID.IsCountry() {
		c = Low
	}
	if tag, err := tt.Maximize(); err == nil && tag.LangID != 0 {
		return Base{tag.LangID}, c
	}
	return Base{0}, No
}

// Script infers the script for the language tag. If it was not explicitly given, it will infer
// a most likely candidate.
// If more than one script is commonly used for a language, the most likely one
// is returned with a low confidence indication. For example, it returns (Cyrl, Low)
// for Serbian.
// If a script cannot be inferred (Zzzz, No) is returned. We do not use Zyyy (undetermined)
// as one would suspect from the IANA registry for BCP 47. In a Unicode context Zyyy marks
// common characters (like 1, 2, 3, '.', etc.) and is therefore more like multiple scripts.
// See https://www.unicode.org/reports/tr24/#Values for more details. Zzzz is also used for
// unknown value in CLDR.  (Zzzz, Exact) is returned if Zzzz was explicitly specified.
// Note that an inferred script is never guaranteed to be the correct one. Latin is
// almost exclusively used for Afrikaans, but Arabic has been used for some texts
// in the past.  Also, the script that is commonly used may change over time.
// It uses a variant of CLDR's Add Likely Subtags algorithm. This is subject to change.
func (t Tag) Script() (Script, Confidence) {
	if scr := t.script(); scr != 0 {
		return Script{scr}, Exact
	}
	tt := t.tag()
	sc, c := language.Script(_Zzzz), No
	if scr := tt.LangID.SuppressScript(); scr != 0 {
		// Note: it is not always the case that a language with a suppress
		// script value is only written in one script (e.g. kk, ms, pa).
		if tt.RegionID == 0 {
			return Script{scr}, High
		}
		sc, c = scr, High
	}
	if tag, err := tt.Maximize(); err == nil {
		if tag.ScriptID != sc {
			sc, c = tag.ScriptID, Low
		}
	} else {
		tt, _ = canonicalize(Deprecated|Macro, tt)
		if tag, err := tt.Maximize(); err == nil && tag.ScriptID != sc {
			sc, c = tag.ScriptID, Low
		}
	}
	return Script{sc}, c
}

// Region returns the region for the language tag. If it was not explicitly given, it will
// infer a most likely candidate from the context.
// It uses a variant of CLDR's Add Likely Subtags algorithm. This is subject to change.
func (t Tag) Region() (Region, Confidence) {
	if r := t.region(); r != 0 {
		return Region{r}, Exact
	}
	tt := t.tag()
	if tt, err := tt.Maximize(); err == nil {
		return Region{tt.RegionID}, Low // TODO: differentiate between high and low.
	}
	tt, _ = canonicalize(Deprecated|Macro, tt)
	if tag, err := tt.Maximize(); err == nil {
		return Region{tag.RegionID}, Low
	}
	return Region{_ZZ}, No // TODO: return world instead of undetermined?
}

// Variants returns the variants specified explicitly for this language tag.
// or nil if no variant was specified.
func (t Tag) Variants() []Variant {
	if !compact.Tag(t).MayHaveVariants() {
		return nil
	}
	v := []Variant{}
	x, str := "", t.tag().Variants()
	for str != "" {
		x, str = nextToken(str)
		v = append(v, Variant{x})
	}
	return v
}

// Parent returns the CLDR parent of t. In CLDR, missing fields in data for a
// specific language are substituted with fields from the parent language.
// The parent for a language may change for newer versions of CLDR.
//
// Parent returns a tag for a less specific language that is mutually
// intelligible or Und if there is no such language. This may not be the same as
// simply stripping the last BCP 47 subtag. For instance, the parent of "zh-TW"
// is "zh-Hant", and the parent of "zh-Hant" is "und".
func (t Tag) Parent() Tag {
	return Tag(compact.Tag(t).Parent())
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

// Extension is a single BCP 47 extension.
type Extension struct {
	s string
}

// String returns the string representation of the extension, including the
// type tag.
func (e Extension) String() string {
	return e.s
}

// ParseExtension parses s as an extension and returns it on success.
func ParseExtension(s string) (e Extension, err error) {
	ext, err := language.ParseExtension(s)
	return Extension{ext}, err
}

// Type returns the one-byte extension type of e. It returns 0 for the zero
// exception.
func (e Extension) Type() byte {
	if e.s == "" {
		return 0
	}
	return e.s[0]
}

// Tokens returns the list of tokens of e.
func (e Extension) Tokens() []string {
	return strings.Split(e.s, "-")
}

// Extension returns the extension of type x for tag t. It will return
// false for ok if t does not have the requested extension. The returned
// extension will be invalid in this case.
func (t Tag) Extension(x byte) (ext Extension, ok bool) {
	if !compact.Tag(t).MayHaveExtensions() {
		return Extension{}, false
	}
	e, ok := t.tag().Extension(x)
	return Extension{e}, ok
}

// Extensions returns all extensions of t.
func (t Tag) Extensions() []Extension {
	if !compact.Tag(t).MayHaveExtensions() {
		return nil
	}
	e := []Extension{}
	for _, ext := range t.tag().Extensions() {
		e = append(e, Extension{ext})
	}
	return e
}

// TypeForKey returns the type associated with the given key, where key and type
// are of the allowed values defined for the Unicode locale extension ('u') in
// https://www.unicode.org/reports/tr35/#Unicode_Language_and_Locale_Identifiers.
// TypeForKey will traverse the inheritance chain to get the correct value.
//
// If there are multiple types associated with a key, only the first will be
// returned. If there is no type associated with a key, it returns the empty
// string.
func (t Tag) TypeForKey(key string) string {
	if !compact.Tag(t).MayHaveExtensions() {
		if key != "rg" && key != "va" {
			return ""
		}
	}
	return t.tag().TypeForKey(key)
}

// SetTypeForKey returns a new Tag with the key set to type, where key and type
// are of the allowed values defined for the Unicode locale extension ('u') in
// https://www.unicode.org/reports/tr35/#Unicode_Language_and_Locale_Identifiers.
// An empty value removes an existing pair with the same key.
func (t Tag) SetTypeForKey(key, value string) (Tag, error) {
	tt, err := t.tag().SetTypeForKey(key, value)
	return makeTag(tt), err
}

// NumCompactTags is the number of compact tags. The maximum tag is
// NumCompactTags-1.
const NumCompactTags = compact.NumCompactTags

// CompactIndex returns an index, where 0 <= index < NumCompactTags, for tags
// for which data exists in the text repository.The index will change over time
// and should not be stored in persistent storage. If t does not match a compact
// index, exact will be false and the compact index will be returned for the
// first match after repeatedly taking the Parent of t.
func CompactIndex(t Tag) (index int, exact bool) {
	id, exact := compact.LanguageID(compact.Tag(t))
	return int(id), exact
}

var root = language.Tag{}

// Base is an ISO 639 language code, used for encoding the base language
// of a language tag.
type Base struct {
	langID language.Language
}

// ParseBase parses a 2- or 3-letter ISO 639 code.
// It returns a ValueError if s is a well-formed but unknown language identifier
// or another error if another error occurred.
func ParseBase(s string) (Base, error) {
	l, err := language.ParseBase(s)
	return Base{l}, err
}

// String returns the BCP 47 representation of the base language.
func (b Base) String() string {
	return b.langID.String()
}

// ISO3 returns the ISO 639-3 language code.
func (b Base) ISO3() string {
	return b.langID.ISO3()
}

// IsPrivateUse reports whether this language code is reserved for private use.
func (b Base) IsPrivateUse() bool {
	return b.langID.IsPrivateUse()
}

// Script is a 4-letter ISO 15924 code for representing scripts.
// It is idiomatically represented in title case.
type Script struct {
	scriptID language.Script
}

// ParseScript parses a 4-letter ISO 15924 code.
// It returns a ValueError if s is a well-formed but unknown script identifier
// or another error if another error occurred.
func ParseScript(s string) (Script, error) {
	sc, err := language.ParseScript(s)
	return Script{sc}, err
}

// String returns the script code in title case.
// It returns "Zzzz" for an unspecified script.
func (s Script) String() string {
	return s.scriptID.String()
}

// IsPrivateUse reports whether this script code is reserved for private use.
func (s Script) IsPrivateUse() bool {
	return s.scriptID.IsPrivateUse()
}

// Region is an ISO 3166-1 or UN M.49 code for representing countries and regions.
type Region struct {
	regionID language.Region
}

// EncodeM49 returns the Region for the given UN M.49 code.
// It returns an error if r is not a valid code.
func EncodeM49(r int) (Region, error) {
	rid, err := language.EncodeM49(r)
	return Region{rid}, err
}

// ParseRegion parses a 2- or 3-letter ISO 3166-1 or a UN M.49 code.
// It returns a ValueError if s is a well-formed but unknown region identifier
// or another error if another error occurred.
func ParseRegion(s string) (Region, error) {
	r, err := language.ParseRegion(s)
	return Region{r}, err
}

// String returns the BCP 47 representation for the region.
// It returns "ZZ" for an unspecified region.
func (r Region) String() string {
	return r.regionID.String()
}

// ISO3 returns the 3-letter ISO code of r.
// Note that not all regions have a 3-letter ISO code.
// In such cases this method returns "ZZZ".
func (r Region) ISO3() string {
	return r.regionID.ISO3()
}

// M49 returns the UN M.49 encoding of r, or 0 if this encoding
// is not defined for r.
func (r Region) M49() int {
	return r.regionID.M49()
}

// IsPrivateUse reports whether r has the ISO 3166 User-assigned status. This
// may include private-use tags that are assigned by CLDR and used in this
// implementation. So IsPrivateUse and IsCountry can be simultaneously true.
func (r Region) IsPrivateUse() bool {
	return r.regionID.IsPrivateUse()
}

// IsCountry returns whether this region is a country or autonomous area. This
// includes non-standard definitions from CLDR.
func (r Region) IsCountry() bool {
	return r.regionID.IsCountry()
}

// IsGroup returns whether this region defines a collection of regions. This
// includes non-standard definitions from CLDR.
func (r Region) IsGroup() bool {
	return r.regionID.IsGroup()
}

// Contains returns whether Region c is contained by Region r. It returns true
// if c == r.
func (r Region) Contains(c Region) bool {
	return r.regionID.Contains(c.regionID)
}

// TLD returns the country code top-level domain (ccTLD). UK is returned for GB.
// In all other cases it returns either the region itself or an error.
//
// This method may return an error for a region for which there exists a
// canonical form with a ccTLD. To get that ccTLD canonicalize r first. The
// region will already be canonicalized it was obtained from a Tag that was
// obtained using any of the default methods.
func (r Region) TLD() (Region, error) {
	tld, err := r.regionID.TLD()
	return Region{tld}, err
}

// Canonicalize returns the region or a possible replacement if the region is
// deprecated. It will not return a replacement for deprecated regions that
// are split into multiple regions.
func (r Region) Canonicalize() Region {
	return Region{r.regionID.Canonicalize()}
}

// Variant represents a registered variant of a language as defined by BCP 47.
type Variant struct {
	variant string
}

// ParseVariant parses and returns a Variant. An error is returned if s is not
// a valid variant.
func ParseVariant(s string) (Variant, error) {
	v, err := language.ParseVariant(s)
	return Variant{v.String()}, err
}

// String returns the string representation of the variant.
func (v Variant) String() string {
	return v.variant
}
