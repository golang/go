// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package norm

// This file contains Form-specific logic and wrappers for data in tables.go.

// Rune info is stored in a separate trie per composing form. A composing form
// and its corresponding decomposing form share the same trie.  Each trie maps
// a rune to a uint16. The values take two forms.  For v >= 0x8000:
//   bits
//   0..8:   ccc
//   9..12:  qcInfo (see below). isYesD is always true (no decompostion).
//   16:     1
// For v < 0x8000, the respective rune has a decomposition and v is an index
// into a byte array of UTF-8 decomposition sequences and additional info and
// has the form:
//    <header> <decomp_byte>* [<tccc> [<lccc>]]
// The header contains the number of bytes in the decomposition (excluding this
// length byte). The two most significant bits of this lenght byte correspond
// to bit 2 and 3 of qcIfo (see below).  The byte sequence itself starts at v+1.
// The byte sequence is followed by a trailing and leading CCC if the values
// for these are not zero.  The value of v determines which ccc are appended
// to the sequences.  For v < firstCCC, there are none, for v >= firstCCC,
// the seqence is followed by a trailing ccc, and for v >= firstLeadingCC
// there is an additional leading ccc.

const (
	qcInfoMask      = 0xF  // to clear all but the relevant bits in a qcInfo
	headerLenMask   = 0x3F // extract the lenght value from the header byte
	headerFlagsMask = 0xC0 // extract the qcInfo bits from the header byte
)

// runeInfo is a representation for the data stored in charinfoTrie.
type runeInfo struct {
	pos   uint8  // start position in reorderBuffer; used in composition.go
	size  uint8  // length of UTF-8 encoding of this rune
	ccc   uint8  // leading canonical combining class (ccc if not decomposition)
	tccc  uint8  // trailing canonical combining class (ccc if not decomposition)
	flags qcInfo // quick check flags
	index uint16
}

// functions dispatchable per form
type lookupFunc func(b input, i int) runeInfo

// formInfo holds Form-specific functions and tables.
type formInfo struct {
	form                     Form
	composing, compatibility bool // form type
	info                     lookupFunc
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
		if Form(i) == NFC || Form(i) == NFKC {
			f.composing = true
		}
	}
}

// We do not distinguish between boundaries for NFC, NFD, etc. to avoid
// unexpected behavior for the user.  For example, in NFD, there is a boundary
// after 'a'.  However, a might combine with modifiers, so from the application's
// perspective it is not a good boundary. We will therefore always use the 
// boundaries for the combining variants.
func (i runeInfo) boundaryBefore() bool {
	if i.ccc == 0 && !i.combinesBackward() {
		return true
	}
	// We assume that the CCC of the first character in a decomposition
	// is always non-zero if different from info.ccc and that we can return
	// false at this point. This is verified by maketables.
	return false
}

func (i runeInfo) boundaryAfter() bool {
	return i.isInert()
}

// We pack quick check data in 4 bits:
//   0:    NFD_QC Yes (0) or No (1). No also means there is a decomposition.
//   1..2: NFC_QC Yes(00), No (10), or Maybe (11)
//   3:    Combines forward  (0 == false, 1 == true)
// 
// When all 4 bits are zero, the character is inert, meaning it is never
// influenced by normalization.
type qcInfo uint8

func (i runeInfo) isYesC() bool { return i.flags&0x4 == 0 }
func (i runeInfo) isYesD() bool { return i.flags&0x1 == 0 }

func (i runeInfo) combinesForward() bool  { return i.flags&0x8 != 0 }
func (i runeInfo) combinesBackward() bool { return i.flags&0x2 != 0 } // == isMaybe
func (i runeInfo) hasDecomposition() bool { return i.flags&0x1 != 0 } // == isNoD

func (r runeInfo) isInert() bool {
	return r.flags&0xf == 0 && r.ccc == 0
}

func (r runeInfo) decomposition() []byte {
	if r.index == 0 {
		return nil
	}
	p := r.index
	n := decomps[p] & 0x3F
	p++
	return decomps[p : p+uint16(n)]
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

func lookupInfoNFC(b input, i int) runeInfo {
	v, sz := b.charinfoNFC(i)
	return compInfo(v, sz)
}

func lookupInfoNFKC(b input, i int) runeInfo {
	v, sz := b.charinfoNFKC(i)
	return compInfo(v, sz)
}

// compInfo converts the information contained in v and sz
// to a runeInfo.  See the comment at the top of the file
// for more information on the format.
func compInfo(v uint16, sz int) runeInfo {
	if v == 0 {
		return runeInfo{size: uint8(sz)}
	} else if v >= 0x8000 {
		return runeInfo{
			size:  uint8(sz),
			ccc:   uint8(v),
			tccc:  uint8(v),
			flags: qcInfo(v>>8) & qcInfoMask,
		}
	}
	// has decomposition
	h := decomps[v]
	f := (qcInfo(h&headerFlagsMask) >> 4) | 0x1
	ri := runeInfo{size: uint8(sz), flags: f, index: v}
	if v >= firstCCC {
		v += uint16(h&headerLenMask) + 1
		ri.tccc = decomps[v]
		if v >= firstLeadingCCC {
			ri.ccc = decomps[v+1]
		}
	}
	return ri
}
