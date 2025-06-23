// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bidi

import "unicode/utf8"

// Properties provides access to BiDi properties of runes.
type Properties struct {
	entry uint8
	last  uint8
}

var trie = newBidiTrie(0)

// TODO: using this for bidirule reduces the running time by about 5%. Consider
// if this is worth exposing or if we can find a way to speed up the Class
// method.
//
// // CompactClass is like Class, but maps all of the BiDi control classes
// // (LRO, RLO, LRE, RLE, PDF, LRI, RLI, FSI, PDI) to the class Control.
// func (p Properties) CompactClass() Class {
// 	return Class(p.entry & 0x0F)
// }

// Class returns the Bidi class for p.
func (p Properties) Class() Class {
	c := Class(p.entry & 0x0F)
	if c == Control {
		c = controlByteToClass[p.last&0xF]
	}
	return c
}

// IsBracket reports whether the rune is a bracket.
func (p Properties) IsBracket() bool { return p.entry&0xF0 != 0 }

// IsOpeningBracket reports whether the rune is an opening bracket.
// IsBracket must return true.
func (p Properties) IsOpeningBracket() bool { return p.entry&openMask != 0 }

// TODO: find a better API and expose.
func (p Properties) reverseBracket(r rune) rune {
	return xorMasks[p.entry>>xorMaskShift] ^ r
}

var controlByteToClass = [16]Class{
	0xD: LRO, // U+202D LeftToRightOverride,
	0xE: RLO, // U+202E RightToLeftOverride,
	0xA: LRE, // U+202A LeftToRightEmbedding,
	0xB: RLE, // U+202B RightToLeftEmbedding,
	0xC: PDF, // U+202C PopDirectionalFormat,
	0x6: LRI, // U+2066 LeftToRightIsolate,
	0x7: RLI, // U+2067 RightToLeftIsolate,
	0x8: FSI, // U+2068 FirstStrongIsolate,
	0x9: PDI, // U+2069 PopDirectionalIsolate,
}

// LookupRune returns properties for r.
func LookupRune(r rune) (p Properties, size int) {
	var buf [4]byte
	n := utf8.EncodeRune(buf[:], r)
	return Lookup(buf[:n])
}

// TODO: these lookup methods are based on the generated trie code. The returned
// sizes have slightly different semantics from the generated code, in that it
// always returns size==1 for an illegal UTF-8 byte (instead of the length
// of the maximum invalid subsequence). Most Transformers, like unicode/norm,
// leave invalid UTF-8 untouched, in which case it has performance benefits to
// do so (without changing the semantics). Bidi requires the semantics used here
// for the bidirule implementation to be compatible with the Go semantics.
//  They ultimately should perhaps be adopted by all trie implementations, for
// convenience sake.
// This unrolled code also boosts performance of the secure/bidirule package by
// about 30%.
// So, to remove this code:
//   - add option to trie generator to define return type.
//   - always return 1 byte size for ill-formed UTF-8 runes.

// Lookup returns properties for the first rune in s and the width in bytes of
// its encoding. The size will be 0 if s does not hold enough bytes to complete
// the encoding.
func Lookup(s []byte) (p Properties, sz int) {
	c0 := s[0]
	switch {
	case c0 < 0x80: // is ASCII
		return Properties{entry: bidiValues[c0]}, 1
	case c0 < 0xC2:
		return Properties{}, 1
	case c0 < 0xE0: // 2-byte UTF-8
		if len(s) < 2 {
			return Properties{}, 0
		}
		i := bidiIndex[c0]
		c1 := s[1]
		if c1 < 0x80 || 0xC0 <= c1 {
			return Properties{}, 1
		}
		return Properties{entry: trie.lookupValue(uint32(i), c1)}, 2
	case c0 < 0xF0: // 3-byte UTF-8
		if len(s) < 3 {
			return Properties{}, 0
		}
		i := bidiIndex[c0]
		c1 := s[1]
		if c1 < 0x80 || 0xC0 <= c1 {
			return Properties{}, 1
		}
		o := uint32(i)<<6 + uint32(c1)
		i = bidiIndex[o]
		c2 := s[2]
		if c2 < 0x80 || 0xC0 <= c2 {
			return Properties{}, 1
		}
		return Properties{entry: trie.lookupValue(uint32(i), c2), last: c2}, 3
	case c0 < 0xF8: // 4-byte UTF-8
		if len(s) < 4 {
			return Properties{}, 0
		}
		i := bidiIndex[c0]
		c1 := s[1]
		if c1 < 0x80 || 0xC0 <= c1 {
			return Properties{}, 1
		}
		o := uint32(i)<<6 + uint32(c1)
		i = bidiIndex[o]
		c2 := s[2]
		if c2 < 0x80 || 0xC0 <= c2 {
			return Properties{}, 1
		}
		o = uint32(i)<<6 + uint32(c2)
		i = bidiIndex[o]
		c3 := s[3]
		if c3 < 0x80 || 0xC0 <= c3 {
			return Properties{}, 1
		}
		return Properties{entry: trie.lookupValue(uint32(i), c3)}, 4
	}
	// Illegal rune
	return Properties{}, 1
}

// LookupString returns properties for the first rune in s and the width in
// bytes of its encoding. The size will be 0 if s does not hold enough bytes to
// complete the encoding.
func LookupString(s string) (p Properties, sz int) {
	c0 := s[0]
	switch {
	case c0 < 0x80: // is ASCII
		return Properties{entry: bidiValues[c0]}, 1
	case c0 < 0xC2:
		return Properties{}, 1
	case c0 < 0xE0: // 2-byte UTF-8
		if len(s) < 2 {
			return Properties{}, 0
		}
		i := bidiIndex[c0]
		c1 := s[1]
		if c1 < 0x80 || 0xC0 <= c1 {
			return Properties{}, 1
		}
		return Properties{entry: trie.lookupValue(uint32(i), c1)}, 2
	case c0 < 0xF0: // 3-byte UTF-8
		if len(s) < 3 {
			return Properties{}, 0
		}
		i := bidiIndex[c0]
		c1 := s[1]
		if c1 < 0x80 || 0xC0 <= c1 {
			return Properties{}, 1
		}
		o := uint32(i)<<6 + uint32(c1)
		i = bidiIndex[o]
		c2 := s[2]
		if c2 < 0x80 || 0xC0 <= c2 {
			return Properties{}, 1
		}
		return Properties{entry: trie.lookupValue(uint32(i), c2), last: c2}, 3
	case c0 < 0xF8: // 4-byte UTF-8
		if len(s) < 4 {
			return Properties{}, 0
		}
		i := bidiIndex[c0]
		c1 := s[1]
		if c1 < 0x80 || 0xC0 <= c1 {
			return Properties{}, 1
		}
		o := uint32(i)<<6 + uint32(c1)
		i = bidiIndex[o]
		c2 := s[2]
		if c2 < 0x80 || 0xC0 <= c2 {
			return Properties{}, 1
		}
		o = uint32(i)<<6 + uint32(c2)
		i = bidiIndex[o]
		c3 := s[3]
		if c3 < 0x80 || 0xC0 <= c3 {
			return Properties{}, 1
		}
		return Properties{entry: trie.lookupValue(uint32(i), c3)}, 4
	}
	// Illegal rune
	return Properties{}, 1
}
