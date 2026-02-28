// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cases

// This file contains the definitions of case mappings for all supported
// languages. The rules for the language-specific tailorings were taken and
// modified from the CLDR transform definitions in common/transforms.

import (
	"strings"
	"unicode"
	"unicode/utf8"

	"golang.org/x/text/internal"
	"golang.org/x/text/language"
	"golang.org/x/text/transform"
	"golang.org/x/text/unicode/norm"
)

// A mapFunc takes a context set to the current rune and writes the mapped
// version to the same context. It may advance the context to the next rune. It
// returns whether a checkpoint is possible: whether the pDst bytes written to
// dst so far won't need changing as we see more source bytes.
type mapFunc func(*context) bool

// A spanFunc takes a context set to the current rune and returns whether this
// rune would be altered when written to the output. It may advance the context
// to the next rune. It returns whether a checkpoint is possible.
type spanFunc func(*context) bool

// maxIgnorable defines the maximum number of ignorables to consider for
// lookahead operations.
const maxIgnorable = 30

// supported lists the language tags for which we have tailorings.
const supported = "und af az el lt nl tr"

func init() {
	tags := []language.Tag{}
	for _, s := range strings.Split(supported, " ") {
		tags = append(tags, language.MustParse(s))
	}
	matcher = internal.NewInheritanceMatcher(tags)
	Supported = language.NewCoverage(tags)
}

var (
	matcher *internal.InheritanceMatcher

	Supported language.Coverage

	// We keep the following lists separate, instead of having a single per-
	// language struct, to give the compiler a chance to remove unused code.

	// Some uppercase mappers are stateless, so we can precompute the
	// Transformers and save a bit on runtime allocations.
	upperFunc = []struct {
		upper mapFunc
		span  spanFunc
	}{
		{nil, nil},                  // und
		{nil, nil},                  // af
		{aztrUpper(upper), isUpper}, // az
		{elUpper, noSpan},           // el
		{ltUpper(upper), noSpan},    // lt
		{nil, nil},                  // nl
		{aztrUpper(upper), isUpper}, // tr
	}

	undUpper            transform.SpanningTransformer = &undUpperCaser{}
	undLower            transform.SpanningTransformer = &undLowerCaser{}
	undLowerIgnoreSigma transform.SpanningTransformer = &undLowerIgnoreSigmaCaser{}

	lowerFunc = []mapFunc{
		nil,       // und
		nil,       // af
		aztrLower, // az
		nil,       // el
		ltLower,   // lt
		nil,       // nl
		aztrLower, // tr
	}

	titleInfos = []struct {
		title     mapFunc
		lower     mapFunc
		titleSpan spanFunc
		rewrite   func(*context)
	}{
		{title, lower, isTitle, nil},                // und
		{title, lower, isTitle, afnlRewrite},        // af
		{aztrUpper(title), aztrLower, isTitle, nil}, // az
		{title, lower, isTitle, nil},                // el
		{ltUpper(title), ltLower, noSpan, nil},      // lt
		{nlTitle, lower, nlTitleSpan, afnlRewrite},  // nl
		{aztrUpper(title), aztrLower, isTitle, nil}, // tr
	}
)

func makeUpper(t language.Tag, o options) transform.SpanningTransformer {
	_, i, _ := matcher.Match(t)
	f := upperFunc[i].upper
	if f == nil {
		return undUpper
	}
	return &simpleCaser{f: f, span: upperFunc[i].span}
}

func makeLower(t language.Tag, o options) transform.SpanningTransformer {
	_, i, _ := matcher.Match(t)
	f := lowerFunc[i]
	if f == nil {
		if o.ignoreFinalSigma {
			return undLowerIgnoreSigma
		}
		return undLower
	}
	if o.ignoreFinalSigma {
		return &simpleCaser{f: f, span: isLower}
	}
	return &lowerCaser{
		first:   f,
		midWord: finalSigma(f),
	}
}

func makeTitle(t language.Tag, o options) transform.SpanningTransformer {
	_, i, _ := matcher.Match(t)
	x := &titleInfos[i]
	lower := x.lower
	if o.noLower {
		lower = (*context).copy
	} else if !o.ignoreFinalSigma {
		lower = finalSigma(lower)
	}
	return &titleCaser{
		title:     x.title,
		lower:     lower,
		titleSpan: x.titleSpan,
		rewrite:   x.rewrite,
	}
}

func noSpan(c *context) bool {
	c.err = transform.ErrEndOfSpan
	return false
}

// TODO: consider a similar special case for the fast majority lower case. This
// is a bit more involved so will require some more precise benchmarking to
// justify it.

type undUpperCaser struct{ transform.NopResetter }

// undUpperCaser implements the Transformer interface for doing an upper case
// mapping for the root locale (und). It eliminates the need for an allocation
// as it prevents escaping by not using function pointers.
func (t undUpperCaser) Transform(dst, src []byte, atEOF bool) (nDst, nSrc int, err error) {
	c := context{dst: dst, src: src, atEOF: atEOF}
	for c.next() {
		upper(&c)
		c.checkpoint()
	}
	return c.ret()
}

func (t undUpperCaser) Span(src []byte, atEOF bool) (n int, err error) {
	c := context{src: src, atEOF: atEOF}
	for c.next() && isUpper(&c) {
		c.checkpoint()
	}
	return c.retSpan()
}

// undLowerIgnoreSigmaCaser implements the Transformer interface for doing
// a lower case mapping for the root locale (und) ignoring final sigma
// handling. This casing algorithm is used in some performance-critical packages
// like secure/precis and x/net/http/idna, which warrants its special-casing.
type undLowerIgnoreSigmaCaser struct{ transform.NopResetter }

func (t undLowerIgnoreSigmaCaser) Transform(dst, src []byte, atEOF bool) (nDst, nSrc int, err error) {
	c := context{dst: dst, src: src, atEOF: atEOF}
	for c.next() && lower(&c) {
		c.checkpoint()
	}
	return c.ret()

}

// Span implements a generic lower-casing. This is possible as isLower works
// for all lowercasing variants. All lowercase variants only vary in how they
// transform a non-lowercase letter. They will never change an already lowercase
// letter. In addition, there is no state.
func (t undLowerIgnoreSigmaCaser) Span(src []byte, atEOF bool) (n int, err error) {
	c := context{src: src, atEOF: atEOF}
	for c.next() && isLower(&c) {
		c.checkpoint()
	}
	return c.retSpan()
}

type simpleCaser struct {
	context
	f    mapFunc
	span spanFunc
}

// simpleCaser implements the Transformer interface for doing a case operation
// on a rune-by-rune basis.
func (t *simpleCaser) Transform(dst, src []byte, atEOF bool) (nDst, nSrc int, err error) {
	c := context{dst: dst, src: src, atEOF: atEOF}
	for c.next() && t.f(&c) {
		c.checkpoint()
	}
	return c.ret()
}

func (t *simpleCaser) Span(src []byte, atEOF bool) (n int, err error) {
	c := context{src: src, atEOF: atEOF}
	for c.next() && t.span(&c) {
		c.checkpoint()
	}
	return c.retSpan()
}

// undLowerCaser implements the Transformer interface for doing a lower case
// mapping for the root locale (und) ignoring final sigma handling. This casing
// algorithm is used in some performance-critical packages like secure/precis
// and x/net/http/idna, which warrants its special-casing.
type undLowerCaser struct{ transform.NopResetter }

func (t undLowerCaser) Transform(dst, src []byte, atEOF bool) (nDst, nSrc int, err error) {
	c := context{dst: dst, src: src, atEOF: atEOF}

	for isInterWord := true; c.next(); {
		if isInterWord {
			if c.info.isCased() {
				if !lower(&c) {
					break
				}
				isInterWord = false
			} else if !c.copy() {
				break
			}
		} else {
			if c.info.isNotCasedAndNotCaseIgnorable() {
				if !c.copy() {
					break
				}
				isInterWord = true
			} else if !c.hasPrefix("Σ") {
				if !lower(&c) {
					break
				}
			} else if !finalSigmaBody(&c) {
				break
			}
		}
		c.checkpoint()
	}
	return c.ret()
}

func (t undLowerCaser) Span(src []byte, atEOF bool) (n int, err error) {
	c := context{src: src, atEOF: atEOF}
	for c.next() && isLower(&c) {
		c.checkpoint()
	}
	return c.retSpan()
}

// lowerCaser implements the Transformer interface. The default Unicode lower
// casing requires different treatment for the first and subsequent characters
// of a word, most notably to handle the Greek final Sigma.
type lowerCaser struct {
	undLowerIgnoreSigmaCaser

	context

	first, midWord mapFunc
}

func (t *lowerCaser) Transform(dst, src []byte, atEOF bool) (nDst, nSrc int, err error) {
	t.context = context{dst: dst, src: src, atEOF: atEOF}
	c := &t.context

	for isInterWord := true; c.next(); {
		if isInterWord {
			if c.info.isCased() {
				if !t.first(c) {
					break
				}
				isInterWord = false
			} else if !c.copy() {
				break
			}
		} else {
			if c.info.isNotCasedAndNotCaseIgnorable() {
				if !c.copy() {
					break
				}
				isInterWord = true
			} else if !t.midWord(c) {
				break
			}
		}
		c.checkpoint()
	}
	return c.ret()
}

// titleCaser implements the Transformer interface. Title casing algorithms
// distinguish between the first letter of a word and subsequent letters of the
// same word. It uses state to avoid requiring a potentially infinite lookahead.
type titleCaser struct {
	context

	// rune mappings used by the actual casing algorithms.
	title     mapFunc
	lower     mapFunc
	titleSpan spanFunc

	rewrite func(*context)
}

// Transform implements the standard Unicode title case algorithm as defined in
// Chapter 3 of The Unicode Standard:
// toTitlecase(X): Find the word boundaries in X according to Unicode Standard
// Annex #29, "Unicode Text Segmentation." For each word boundary, find the
// first cased character F following the word boundary. If F exists, map F to
// Titlecase_Mapping(F); then map all characters C between F and the following
// word boundary to Lowercase_Mapping(C).
func (t *titleCaser) Transform(dst, src []byte, atEOF bool) (nDst, nSrc int, err error) {
	t.context = context{dst: dst, src: src, atEOF: atEOF, isMidWord: t.isMidWord}
	c := &t.context

	if !c.next() {
		return c.ret()
	}

	for {
		p := c.info
		if t.rewrite != nil {
			t.rewrite(c)
		}

		wasMid := p.isMid()
		// Break out of this loop on failure to ensure we do not modify the
		// state incorrectly.
		if p.isCased() {
			if !c.isMidWord {
				if !t.title(c) {
					break
				}
				c.isMidWord = true
			} else if !t.lower(c) {
				break
			}
		} else if !c.copy() {
			break
		} else if p.isBreak() {
			c.isMidWord = false
		}

		// As we save the state of the transformer, it is safe to call
		// checkpoint after any successful write.
		if !(c.isMidWord && wasMid) {
			c.checkpoint()
		}

		if !c.next() {
			break
		}
		if wasMid && c.info.isMid() {
			c.isMidWord = false
		}
	}
	return c.ret()
}

func (t *titleCaser) Span(src []byte, atEOF bool) (n int, err error) {
	t.context = context{src: src, atEOF: atEOF, isMidWord: t.isMidWord}
	c := &t.context

	if !c.next() {
		return c.retSpan()
	}

	for {
		p := c.info
		if t.rewrite != nil {
			t.rewrite(c)
		}

		wasMid := p.isMid()
		// Break out of this loop on failure to ensure we do not modify the
		// state incorrectly.
		if p.isCased() {
			if !c.isMidWord {
				if !t.titleSpan(c) {
					break
				}
				c.isMidWord = true
			} else if !isLower(c) {
				break
			}
		} else if p.isBreak() {
			c.isMidWord = false
		}
		// As we save the state of the transformer, it is safe to call
		// checkpoint after any successful write.
		if !(c.isMidWord && wasMid) {
			c.checkpoint()
		}

		if !c.next() {
			break
		}
		if wasMid && c.info.isMid() {
			c.isMidWord = false
		}
	}
	return c.retSpan()
}

// finalSigma adds Greek final Sigma handing to another casing function. It
// determines whether a lowercased sigma should be σ or ς, by looking ahead for
// case-ignorables and a cased letters.
func finalSigma(f mapFunc) mapFunc {
	return func(c *context) bool {
		if !c.hasPrefix("Σ") {
			return f(c)
		}
		return finalSigmaBody(c)
	}
}

func finalSigmaBody(c *context) bool {
	// Current rune must be ∑.

	// ::NFD();
	// # 03A3; 03C2; 03A3; 03A3; Final_Sigma; # GREEK CAPITAL LETTER SIGMA
	// Σ } [:case-ignorable:]* [:cased:] → σ;
	// [:cased:] [:case-ignorable:]* { Σ → ς;
	// ::Any-Lower;
	// ::NFC();

	p := c.pDst
	c.writeString("ς")

	// TODO: we should do this here, but right now this will never have an
	// effect as this is called when the prefix is Sigma, whereas Dutch and
	// Afrikaans only test for an apostrophe.
	//
	// if t.rewrite != nil {
	// 	t.rewrite(c)
	// }

	// We need to do one more iteration after maxIgnorable, as a cased
	// letter is not an ignorable and may modify the result.
	wasMid := false
	for i := 0; i < maxIgnorable+1; i++ {
		if !c.next() {
			return false
		}
		if !c.info.isCaseIgnorable() {
			// All Midword runes are also case ignorable, so we are
			// guaranteed to have a letter or word break here. As we are
			// unreading the run, there is no need to unset c.isMidWord;
			// the title caser will handle this.
			if c.info.isCased() {
				// p+1 is guaranteed to be in bounds: if writing ς was
				// successful, p+1 will contain the second byte of ς. If not,
				// this function will have returned after c.next returned false.
				c.dst[p+1]++ // ς → σ
			}
			c.unreadRune()
			return true
		}
		// A case ignorable may also introduce a word break, so we may need
		// to continue searching even after detecting a break.
		isMid := c.info.isMid()
		if (wasMid && isMid) || c.info.isBreak() {
			c.isMidWord = false
		}
		wasMid = isMid
		c.copy()
	}
	return true
}

// finalSigmaSpan would be the same as isLower.

// elUpper implements Greek upper casing, which entails removing a predefined
// set of non-blocked modifiers. Note that these accents should not be removed
// for title casing!
// Example: "Οδός" -> "ΟΔΟΣ".
func elUpper(c *context) bool {
	// From CLDR:
	// [:Greek:] [^[:ccc=Not_Reordered:][:ccc=Above:]]*? { [\u0313\u0314\u0301\u0300\u0306\u0342\u0308\u0304] → ;
	// [:Greek:] [^[:ccc=Not_Reordered:][:ccc=Iota_Subscript:]]*? { \u0345 → ;

	r, _ := utf8.DecodeRune(c.src[c.pSrc:])
	oldPDst := c.pDst
	if !upper(c) {
		return false
	}
	if !unicode.Is(unicode.Greek, r) {
		return true
	}
	i := 0
	// Take the properties of the uppercased rune that is already written to the
	// destination. This saves us the trouble of having to uppercase the
	// decomposed rune again.
	if b := norm.NFD.Properties(c.dst[oldPDst:]).Decomposition(); b != nil {
		// Restore the destination position and process the decomposed rune.
		r, sz := utf8.DecodeRune(b)
		if r <= 0xFF { // See A.6.1
			return true
		}
		c.pDst = oldPDst
		// Insert the first rune and ignore the modifiers. See A.6.2.
		c.writeBytes(b[:sz])
		i = len(b[sz:]) / 2 // Greek modifiers are always of length 2.
	}

	for ; i < maxIgnorable && c.next(); i++ {
		switch r, _ := utf8.DecodeRune(c.src[c.pSrc:]); r {
		// Above and Iota Subscript
		case 0x0300, // U+0300 COMBINING GRAVE ACCENT
			0x0301, // U+0301 COMBINING ACUTE ACCENT
			0x0304, // U+0304 COMBINING MACRON
			0x0306, // U+0306 COMBINING BREVE
			0x0308, // U+0308 COMBINING DIAERESIS
			0x0313, // U+0313 COMBINING COMMA ABOVE
			0x0314, // U+0314 COMBINING REVERSED COMMA ABOVE
			0x0342, // U+0342 COMBINING GREEK PERISPOMENI
			0x0345: // U+0345 COMBINING GREEK YPOGEGRAMMENI
			// No-op. Gobble the modifier.

		default:
			switch v, _ := trie.lookup(c.src[c.pSrc:]); info(v).cccType() {
			case cccZero:
				c.unreadRune()
				return true

			// We don't need to test for IotaSubscript as the only rune that
			// qualifies (U+0345) was already excluded in the switch statement
			// above. See A.4.

			case cccAbove:
				return c.copy()
			default:
				// Some other modifier. We're still allowed to gobble Greek
				// modifiers after this.
				c.copy()
			}
		}
	}
	return i == maxIgnorable
}

// TODO: implement elUpperSpan (low-priority: complex and infrequent).

func ltLower(c *context) bool {
	// From CLDR:
	// # Introduce an explicit dot above when lowercasing capital I's and J's
	// # whenever there are more accents above.
	// # (of the accents used in Lithuanian: grave, acute, tilde above, and ogonek)
	// # 0049; 0069 0307; 0049; 0049; lt More_Above; # LATIN CAPITAL LETTER I
	// # 004A; 006A 0307; 004A; 004A; lt More_Above; # LATIN CAPITAL LETTER J
	// # 012E; 012F 0307; 012E; 012E; lt More_Above; # LATIN CAPITAL LETTER I WITH OGONEK
	// # 00CC; 0069 0307 0300; 00CC; 00CC; lt; # LATIN CAPITAL LETTER I WITH GRAVE
	// # 00CD; 0069 0307 0301; 00CD; 00CD; lt; # LATIN CAPITAL LETTER I WITH ACUTE
	// # 0128; 0069 0307 0303; 0128; 0128; lt; # LATIN CAPITAL LETTER I WITH TILDE
	// ::NFD();
	// I } [^[:ccc=Not_Reordered:][:ccc=Above:]]* [:ccc=Above:] → i \u0307;
	// J } [^[:ccc=Not_Reordered:][:ccc=Above:]]* [:ccc=Above:] → j \u0307;
	// I \u0328 (Į) } [^[:ccc=Not_Reordered:][:ccc=Above:]]* [:ccc=Above:] → i \u0328 \u0307;
	// I \u0300 (Ì) → i \u0307 \u0300;
	// I \u0301 (Í) → i \u0307 \u0301;
	// I \u0303 (Ĩ) → i \u0307 \u0303;
	// ::Any-Lower();
	// ::NFC();

	i := 0
	if r := c.src[c.pSrc]; r < utf8.RuneSelf {
		lower(c)
		if r != 'I' && r != 'J' {
			return true
		}
	} else {
		p := norm.NFD.Properties(c.src[c.pSrc:])
		if d := p.Decomposition(); len(d) >= 3 && (d[0] == 'I' || d[0] == 'J') {
			// UTF-8 optimization: the decomposition will only have an above
			// modifier if the last rune of the decomposition is in [U+300-U+311].
			// In all other cases, a decomposition starting with I is always
			// an I followed by modifiers that are not cased themselves. See A.2.
			if d[1] == 0xCC && d[2] <= 0x91 { // A.2.4.
				if !c.writeBytes(d[:1]) {
					return false
				}
				c.dst[c.pDst-1] += 'a' - 'A' // lower

				// Assumption: modifier never changes on lowercase. See A.1.
				// Assumption: all modifiers added have CCC = Above. See A.2.3.
				return c.writeString("\u0307") && c.writeBytes(d[1:])
			}
			// In all other cases the additional modifiers will have a CCC
			// that is less than 230 (Above). We will insert the U+0307, if
			// needed, after these modifiers so that a string in FCD form
			// will remain so. See A.2.2.
			lower(c)
			i = 1
		} else {
			return lower(c)
		}
	}

	for ; i < maxIgnorable && c.next(); i++ {
		switch c.info.cccType() {
		case cccZero:
			c.unreadRune()
			return true
		case cccAbove:
			return c.writeString("\u0307") && c.copy() // See A.1.
		default:
			c.copy() // See A.1.
		}
	}
	return i == maxIgnorable
}

// ltLowerSpan would be the same as isLower.

func ltUpper(f mapFunc) mapFunc {
	return func(c *context) bool {
		// Unicode:
		// 0307; 0307; ; ; lt After_Soft_Dotted; # COMBINING DOT ABOVE
		//
		// From CLDR:
		// # Remove \u0307 following soft-dotteds (i, j, and the like), with possible
		// # intervening non-230 marks.
		// ::NFD();
		// [:Soft_Dotted:] [^[:ccc=Not_Reordered:][:ccc=Above:]]* { \u0307 → ;
		// ::Any-Upper();
		// ::NFC();

		// TODO: See A.5. A soft-dotted rune never has an exception. This would
		// allow us to overload the exception bit and encode this property in
		// info. Need to measure performance impact of this.
		r, _ := utf8.DecodeRune(c.src[c.pSrc:])
		oldPDst := c.pDst
		if !f(c) {
			return false
		}
		if !unicode.Is(unicode.Soft_Dotted, r) {
			return true
		}

		// We don't need to do an NFD normalization, as a soft-dotted rune never
		// contains U+0307. See A.3.

		i := 0
		for ; i < maxIgnorable && c.next(); i++ {
			switch c.info.cccType() {
			case cccZero:
				c.unreadRune()
				return true
			case cccAbove:
				if c.hasPrefix("\u0307") {
					// We don't do a full NFC, but rather combine runes for
					// some of the common cases. (Returning NFC or
					// preserving normal form is neither a requirement nor
					// a possibility anyway).
					if !c.next() {
						return false
					}
					if c.dst[oldPDst] == 'I' && c.pDst == oldPDst+1 && c.src[c.pSrc] == 0xcc {
						s := ""
						switch c.src[c.pSrc+1] {
						case 0x80: // U+0300 COMBINING GRAVE ACCENT
							s = "\u00cc" // U+00CC LATIN CAPITAL LETTER I WITH GRAVE
						case 0x81: // U+0301 COMBINING ACUTE ACCENT
							s = "\u00cd" // U+00CD LATIN CAPITAL LETTER I WITH ACUTE
						case 0x83: // U+0303 COMBINING TILDE
							s = "\u0128" // U+0128 LATIN CAPITAL LETTER I WITH TILDE
						case 0x88: // U+0308 COMBINING DIAERESIS
							s = "\u00cf" // U+00CF LATIN CAPITAL LETTER I WITH DIAERESIS
						default:
						}
						if s != "" {
							c.pDst = oldPDst
							return c.writeString(s)
						}
					}
				}
				return c.copy()
			default:
				c.copy()
			}
		}
		return i == maxIgnorable
	}
}

// TODO: implement ltUpperSpan (low priority: complex and infrequent).

func aztrUpper(f mapFunc) mapFunc {
	return func(c *context) bool {
		// i→İ;
		if c.src[c.pSrc] == 'i' {
			return c.writeString("İ")
		}
		return f(c)
	}
}

func aztrLower(c *context) (done bool) {
	// From CLDR:
	// # I and i-dotless; I-dot and i are case pairs in Turkish and Azeri
	// # 0130; 0069; 0130; 0130; tr; # LATIN CAPITAL LETTER I WITH DOT ABOVE
	// İ→i;
	// # When lowercasing, remove dot_above in the sequence I + dot_above, which will turn into i.
	// # This matches the behavior of the canonically equivalent I-dot_above
	// # 0307; ; 0307; 0307; tr After_I; # COMBINING DOT ABOVE
	// # When lowercasing, unless an I is before a dot_above, it turns into a dotless i.
	// # 0049; 0131; 0049; 0049; tr Not_Before_Dot; # LATIN CAPITAL LETTER I
	// I([^[:ccc=Not_Reordered:][:ccc=Above:]]*)\u0307 → i$1 ;
	// I→ı ;
	// ::Any-Lower();
	if c.hasPrefix("\u0130") { // İ
		return c.writeString("i")
	}
	if c.src[c.pSrc] != 'I' {
		return lower(c)
	}

	// We ignore the lower-case I for now, but insert it later when we know
	// which form we need.
	start := c.pSrc + c.sz

	i := 0
Loop:
	// We check for up to n ignorables before \u0307. As \u0307 is an
	// ignorable as well, n is maxIgnorable-1.
	for ; i < maxIgnorable && c.next(); i++ {
		switch c.info.cccType() {
		case cccAbove:
			if c.hasPrefix("\u0307") {
				return c.writeString("i") && c.writeBytes(c.src[start:c.pSrc]) // ignore U+0307
			}
			done = true
			break Loop
		case cccZero:
			c.unreadRune()
			done = true
			break Loop
		default:
			// We'll write this rune after we know which starter to use.
		}
	}
	if i == maxIgnorable {
		done = true
	}
	return c.writeString("ı") && c.writeBytes(c.src[start:c.pSrc+c.sz]) && done
}

// aztrLowerSpan would be the same as isLower.

func nlTitle(c *context) bool {
	// From CLDR:
	// # Special titlecasing for Dutch initial "ij".
	// ::Any-Title();
	// # Fix up Ij at the beginning of a "word" (per Any-Title, notUAX #29)
	// [:^WB=ALetter:] [:WB=Extend:]* [[:WB=MidLetter:][:WB=MidNumLet:]]? { Ij } → IJ ;
	if c.src[c.pSrc] != 'I' && c.src[c.pSrc] != 'i' {
		return title(c)
	}

	if !c.writeString("I") || !c.next() {
		return false
	}
	if c.src[c.pSrc] == 'j' || c.src[c.pSrc] == 'J' {
		return c.writeString("J")
	}
	c.unreadRune()
	return true
}

func nlTitleSpan(c *context) bool {
	// From CLDR:
	// # Special titlecasing for Dutch initial "ij".
	// ::Any-Title();
	// # Fix up Ij at the beginning of a "word" (per Any-Title, notUAX #29)
	// [:^WB=ALetter:] [:WB=Extend:]* [[:WB=MidLetter:][:WB=MidNumLet:]]? { Ij } → IJ ;
	if c.src[c.pSrc] != 'I' {
		return isTitle(c)
	}
	if !c.next() || c.src[c.pSrc] == 'j' {
		return false
	}
	if c.src[c.pSrc] != 'J' {
		c.unreadRune()
	}
	return true
}

// Not part of CLDR, but see https://unicode.org/cldr/trac/ticket/7078.
func afnlRewrite(c *context) {
	if c.hasPrefix("'") || c.hasPrefix("’") {
		c.isMidWord = true
	}
}
