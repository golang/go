// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package norm

// This file contains Form-specific logic and wrappers for data in tables.go.

// Rune info is stored in a separate trie per composing form. A composing form
// and its corresponding decomposing form share the same trie.  Each trie maps
// a rune to a uint16. The values take two forms.  For v >= 0x8000:
//   bits
//   15:    1 (inverse of NFD_QD bit of qcInfo)
//   13..7: qcInfo (see below). isYesD is always true (no decompostion).
//    6..0: ccc (compressed CCC value).
// For v < 0x8000, the respective rune has a decomposition and v is an index
// into a byte array of UTF-8 decomposition sequences and additional info and
// has the form:
//    <header> <decomp_byte>* [<tccc> [<lccc>]]
// The header contains the number of bytes in the decomposition (excluding this
// length byte). The two most significant bits of this length byte correspond
// to bit 5 and 4 of qcInfo (see below).  The byte sequence itself starts at v+1.
// The byte sequence is followed by a trailing and leading CCC if the values
// for these are not zero.  The value of v determines which ccc are appended
// to the sequences.  For v < firstCCC, there are none, for v >= firstCCC,
// the sequence is followed by a trailing ccc, and for v >= firstLeadingCC
// there is an additional leading ccc. The value of tccc itself is the
// trailing CCC shifted left 2 bits. The two least-significant bits of tccc
// are the number of trailing non-starters.

const (
	qcInfoMask      = 0x3F // to clear all but the relevant bits in a qcInfo
	headerLenMask   = 0x3F // extract the length value from the header byte
	headerFlagsMask = 0xC0 // extract the qcInfo bits from the header byte
)

// Properties provides access to normalization properties of a rune.
type Properties struct {
	pos   uint8  // start position in reorderBuffer; used in composition.go
	size  uint8  // length of UTF-8 encoding of this rune
	ccc   uint8  // leading canonical combining class (ccc if not decomposition)
	tccc  uint8  // trailing canonical combining class (ccc if not decomposition)
	nLead uint8  // number of leading non-starters.
	flags qcInfo // quick check flags
	index uint16
}

// functions dispatchable per form
type lookupFunc func(b input, i int) Properties

// formInfo holds Form-specific functions and tables.
type formInfo struct {
	form                     Form
	composing, compatibility bool // form type
	info                     lookupFunc
	nextMain                 iterFunc
}

var formTable []*formInfo

func init() {
	formTable = make([]*formInfo, 4)

	for i := range formTable {
		f := &formInfo{}
		formTable[i] = f
		f.form = Form(i)
		if Form(i) == NFKD || Form(i) == NFKC {
			f.compatibility = true
			f.info = lookupInfoNFKC
		} else {
			f.info = lookupInfoNFC
		}
		f.nextMain = nextDecomposed
		if Form(i) == NFC || Form(i) == NFKC {
			f.nextMain = nextComposed
			f.composing = true
		}
	}
}

// We do not distinguish between boundaries for NFC, NFD, etc. to avoid
// unexpected behavior for the user.  For example, in NFD, there is a boundary
// after 'a'.  However, 'a' might combine with modifiers, so from the application's
// perspective it is not a good boundary. We will therefore always use the
// boundaries for the combining variants.

// BoundaryBefore returns true if this rune starts a new segment and
// cannot combine with any rune on the left.
func (p Properties) BoundaryBefore() bool {
	if p.ccc == 0 && !p.combinesBackward() {
		return true
	}
	// We assume that the CCC of the first character in a decomposition
	// is always non-zero if different from info.ccc and that we can return
	// false at this point. This is verified by maketables.
	return false
}

// BoundaryAfter returns true if runes cannot combine with or otherwise
// interact with this or previous runes.
func (p Properties) BoundaryAfter() bool {
	// TODO: loosen these conditions.
	return p.isInert()
}

// We pack quick check data in 4 bits:
//   5:    Combines forward  (0 == false, 1 == true)
//   4..3: NFC_QC Yes(00), No (10), or Maybe (11)
//   2:    NFD_QC Yes (0) or No (1). No also means there is a decomposition.
//   1..0: Number of trailing non-starters.
//
// When all 4 bits are zero, the character is inert, meaning it is never
// influenced by normalization.
type qcInfo uint8

func (p Properties) isYesC() bool { return p.flags&0x10 == 0 }
func (p Properties) isYesD() bool { return p.flags&0x4 == 0 }

func (p Properties) combinesForward() bool  { return p.flags&0x20 != 0 }
func (p Properties) combinesBackward() bool { return p.flags&0x8 != 0 } // == isMaybe
func (p Properties) hasDecomposition() bool { return p.flags&0x4 != 0 } // == isNoD

func (p Properties) isInert() bool {
	return p.flags&qcInfoMask == 0 && p.ccc == 0
}

func (p Properties) multiSegment() bool {
	return p.index >= firstMulti && p.index < endMulti
}

func (p Properties) nLeadingNonStarters() uint8 {
	return p.nLead
}

func (p Properties) nTrailingNonStarters() uint8 {
	return uint8(p.flags & 0x03)
}

// Decomposition returns the decomposition for the underlying rune
// or nil if there is none.
func (p Properties) Decomposition() []byte {
	// TODO: create the decomposition for Hangul?
	if p.index == 0 {
		return nil
	}
	i := p.index
	n := decomps[i] & headerLenMask
	i++
	return decomps[i : i+uint16(n)]
}

// Size returns the length of UTF-8 encoding of the rune.
func (p Properties) Size() int {
	return int(p.size)
}

// CCC returns the canonical combining class of the underlying rune.
func (p Properties) CCC() uint8 {
	if p.index >= firstCCCZeroExcept {
		return 0
	}
	return ccc[p.ccc]
}

// LeadCCC returns the CCC of the first rune in the decomposition.
// If there is no decomposition, LeadCCC equals CCC.
func (p Properties) LeadCCC() uint8 {
	return ccc[p.ccc]
}

// TrailCCC returns the CCC of the last rune in the decomposition.
// If there is no decomposition, TrailCCC equals CCC.
func (p Properties) TrailCCC() uint8 {
	return ccc[p.tccc]
}

// Recomposition
// We use 32-bit keys instead of 64-bit for the two codepoint keys.
// This clips off the bits of three entries, but we know this will not
// result in a collision. In the unlikely event that changes to
// UnicodeData.txt introduce collisions, the compiler will catch it.
// Note that the recomposition map for NFC and NFKC are identical.

// combine returns the combined rune or 0 if it doesn't exist.
func combine(a, b rune) rune {
	key := uint32(uint16(a))<<16 + uint32(uint16(b))
	return recompMap[key]
}

func lookupInfoNFC(b input, i int) Properties {
	v, sz := b.charinfoNFC(i)
	return compInfo(v, sz)
}

func lookupInfoNFKC(b input, i int) Properties {
	v, sz := b.charinfoNFKC(i)
	return compInfo(v, sz)
}

// Properties returns properties for the first rune in s.
func (f Form) Properties(s []byte) Properties {
	if f == NFC || f == NFD {
		return compInfo(nfcData.lookup(s))
	}
	return compInfo(nfkcData.lookup(s))
}

// PropertiesString returns properties for the first rune in s.
func (f Form) PropertiesString(s string) Properties {
	if f == NFC || f == NFD {
		return compInfo(nfcData.lookupString(s))
	}
	return compInfo(nfkcData.lookupString(s))
}

// compInfo converts the information contained in v and sz
// to a Properties.  See the comment at the top of the file
// for more information on the format.
func compInfo(v uint16, sz int) Properties {
	if v == 0 {
		return Properties{size: uint8(sz)}
	} else if v >= 0x8000 {
		p := Properties{
			size:  uint8(sz),
			ccc:   uint8(v),
			tccc:  uint8(v),
			flags: qcInfo(v >> 8),
		}
		if p.ccc > 0 || p.combinesBackward() {
			p.nLead = uint8(p.flags & 0x3)
		}
		return p
	}
	// has decomposition
	h := decomps[v]
	f := (qcInfo(h&headerFlagsMask) >> 2) | 0x4
	p := Properties{size: uint8(sz), flags: f, index: v}
	if v >= firstCCC {
		v += uint16(h&headerLenMask) + 1
		c := decomps[v]
		p.tccc = c >> 2
		p.flags |= qcInfo(c & 0x3)
		if v >= firstLeadingCCC {
			p.nLead = c & 0x3
			if v >= firstStarterWithNLead {
				// We were tricked. Remove the decomposition.
				p.flags &= 0x03
				p.index = 0
				return p
			}
			p.ccc = decomps[v+1]
		}
	}
	return p
}
