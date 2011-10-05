// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package norm

// This file contains Form-specific logic and wrappers for data in tables.go.

type runeInfo struct {
	pos   uint8  // start position in reorderBuffer; used in composition.go
	size  uint8  // length of UTF-8 encoding of this rune
	ccc   uint8  // canonical combining class
	flags qcInfo // quick check flags
}

// functions dispatchable per form
type boundaryFunc func(f *formInfo, info runeInfo) bool
type lookupFunc func(b input, i int) runeInfo
type decompFunc func(b input, i int) []byte

// formInfo holds Form-specific functions and tables.
type formInfo struct {
	form Form

	composing, compatibility bool // form type

	decompose      decompFunc
	info           lookupFunc
	boundaryBefore boundaryFunc
	boundaryAfter  boundaryFunc
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
			f.decompose = decomposeNFKC
			f.info = lookupInfoNFKC
		} else {
			f.decompose = decomposeNFC
			f.info = lookupInfoNFC
		}
		if Form(i) == NFC || Form(i) == NFKC {
			f.composing = true
			f.boundaryBefore = compBoundaryBefore
			f.boundaryAfter = compBoundaryAfter
		} else {
			f.boundaryBefore = decompBoundary
			f.boundaryAfter = decompBoundary
		}
	}
}

func decompBoundary(f *formInfo, info runeInfo) bool {
	if info.ccc == 0 && info.flags.isYesD() { // Implies isHangul(b) == true
		return true
	}
	// We assume that the CCC of the first character in a decomposition
	// is always non-zero if different from info.ccc and that we can return
	// false at this point. This is verified by maketables.
	return false
}

func compBoundaryBefore(f *formInfo, info runeInfo) bool {
	if info.ccc == 0 && !info.flags.combinesBackward() {
		return true
	}
	// We assume that the CCC of the first character in a decomposition
	// is always non-zero if different from info.ccc and that we can return
	// false at this point. This is verified by maketables.
	return false
}

func compBoundaryAfter(f *formInfo, info runeInfo) bool {
	// This misses values where the last char in a decomposition is a
	// boundary such as Hangul with JamoT.
	return info.isInert()
}

// We pack quick check data in 4 bits:
//   0:    NFD_QC Yes (0) or No (1). No also means there is a decomposition.
//   1..2: NFC_QC Yes(00), No (01), or Maybe (11)
//   3:    Combines forward  (0 == false, 1 == true)
// 
// When all 4 bits are zero, the character is inert, meaning it is never
// influenced by normalization.
//
// We pack the bits for both NFC/D and NFKC/D in one byte.
type qcInfo uint8

func (i qcInfo) isYesC() bool  { return i&0x2 == 0 }
func (i qcInfo) isNoC() bool   { return i&0x6 == 0x2 }
func (i qcInfo) isMaybe() bool { return i&0x4 != 0 }
func (i qcInfo) isYesD() bool  { return i&0x1 == 0 }
func (i qcInfo) isNoD() bool   { return i&0x1 != 0 }

func (i qcInfo) combinesForward() bool  { return i&0x8 != 0 }
func (i qcInfo) combinesBackward() bool { return i&0x4 != 0 } // == isMaybe
func (i qcInfo) hasDecomposition() bool { return i&0x1 != 0 } // == isNoD

func (r runeInfo) isInert() bool {
	return r.flags&0xf == 0 && r.ccc == 0
}

// Wrappers for tables.go

// The 16-bit value of the decompostion tries is an index into a byte
// array of UTF-8 decomposition sequences. The first byte is the number
// of bytes in the decomposition (excluding this length byte). The actual
// sequence starts at the offset+1.
func decomposeNFC(s input, i int) []byte {
	p := s.decomposeNFC(i)
	n := decomps[p]
	p++
	return decomps[p : p+uint16(n)]
}

func decomposeNFKC(s input, i int) []byte {
	p := s.decomposeNFKC(i)
	n := decomps[p]
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
func combine(a, b uint32) uint32 {
	key := uint32(uint16(a))<<16 + uint32(uint16(b))
	return recompMap[key]
}

// The 16-bit character info has the following bit layout:
//    0..7   CCC value.
//    8..11  qcInfo for NFC/NFD
//   12..15  qcInfo for NFKC/NFKD
func lookupInfoNFC(b input, i int) runeInfo {
	v, sz := b.charinfo(i)
	return runeInfo{0, uint8(sz), uint8(v), qcInfo(v >> 8)}
}

func lookupInfoNFKC(b input, i int) runeInfo {
	v, sz := b.charinfo(i)
	return runeInfo{0, uint8(sz), uint8(v), qcInfo(v >> 12)}
}
