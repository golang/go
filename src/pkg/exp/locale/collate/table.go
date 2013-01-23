// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package collate

import (
	"exp/norm"
	"unicode/utf8"
)

// tableIndex holds information for constructing a table
// for a certain locale based on the main table.
type tableIndex struct {
	lookupOffset uint32
	valuesOffset uint32
}

// table holds all collation data for a given collation ordering.
type table struct {
	index trie // main trie

	// expansion info
	expandElem []uint32

	// contraction info
	contractTries  contractTrieSet
	contractElem   []uint32
	maxContractLen int
	variableTop    uint32
}

func (t *table) indexedTable(idx tableIndex) *table {
	nt := *t
	nt.index.index0 = t.index.index[idx.lookupOffset*blockSize:]
	nt.index.values0 = t.index.values[idx.valuesOffset*blockSize:]
	return &nt
}

func (t *table) AppendNext(w []Elem, b []byte) (res []Elem, n int) {
	return t.appendNext(w, source{bytes: b})
}

func (t *table) AppendNextString(w []Elem, s string) (res []Elem, n int) {
	return t.appendNext(w, source{str: s})
}

func (t *table) Start(p int, b []byte) int {
	// TODO: implement
	panic("not implemented")
}

func (t *table) StartString(p int, s string) int {
	// TODO: implement
	panic("not implemented")
}

func (t *table) Domain() []string {
	// TODO: implement
	panic("not implemented")
}

type source struct {
	str   string
	bytes []byte
}

func (src *source) lookup(t *table) (ce Elem, sz int) {
	if src.bytes == nil {
		return t.index.lookupString(src.str)
	}
	return t.index.lookup(src.bytes)
}

func (src *source) tail(sz int) {
	if src.bytes == nil {
		src.str = src.str[sz:]
	} else {
		src.bytes = src.bytes[sz:]
	}
}

func (src *source) nfd(buf []byte, end int) []byte {
	if src.bytes == nil {
		return norm.NFD.AppendString(buf[:0], src.str[:end])
	}
	return norm.NFD.Append(buf[:0], src.bytes[:end]...)
}

func (src *source) rune() (r rune, sz int) {
	if src.bytes == nil {
		return utf8.DecodeRuneInString(src.str)
	}
	return utf8.DecodeRune(src.bytes)
}

func (src *source) properties(f norm.Form) norm.Properties {
	if src.bytes == nil {
		return f.PropertiesString(src.str)
	}
	return f.Properties(src.bytes)
}

// appendNext appends the weights corresponding to the next rune or
// contraction in s.  If a contraction is matched to a discontinuous
// sequence of runes, the weights for the interstitial runes are
// appended as well.  It returns a new slice that includes the appended
// weights and the number of bytes consumed from s.
func (t *table) appendNext(w []Elem, src source) (res []Elem, n int) {
	ce, sz := src.lookup(t)
	tp := ce.ctype()
	if tp == ceNormal {
		if ce == 0 {
			r, _ := src.rune()
			const (
				hangulSize  = 3
				firstHangul = 0xAC00
				lastHangul  = 0xD7A3
			)
			if r >= firstHangul && r <= lastHangul {
				// TODO: performance can be considerably improved here.
				n = sz
				var buf [16]byte // Used for decomposing Hangul.
				for b := src.nfd(buf[:0], hangulSize); len(b) > 0; b = b[sz:] {
					ce, sz = t.index.lookup(b)
					w = append(w, ce)
				}
				return w, n
			}
			ce = makeImplicitCE(implicitPrimary(r))
		}
		w = append(w, ce)
	} else if tp == ceExpansionIndex {
		w = t.appendExpansion(w, ce)
	} else if tp == ceContractionIndex {
		n := 0
		src.tail(sz)
		if src.bytes == nil {
			w, n = t.matchContractionString(w, ce, src.str)
		} else {
			w, n = t.matchContraction(w, ce, src.bytes)
		}
		sz += n
	} else if tp == ceDecompose {
		// Decompose using NFKD and replace tertiary weights.
		t1, t2 := splitDecompose(ce)
		i := len(w)
		nfkd := src.properties(norm.NFKD).Decomposition()
		for p := 0; len(nfkd) > 0; nfkd = nfkd[p:] {
			w, p = t.appendNext(w, source{bytes: nfkd})
		}
		w[i] = w[i].updateTertiary(t1)
		if i++; i < len(w) {
			w[i] = w[i].updateTertiary(t2)
			for i++; i < len(w); i++ {
				w[i] = w[i].updateTertiary(maxTertiary)
			}
		}
	}
	return w, sz
}

func (t *table) appendExpansion(w []Elem, ce Elem) []Elem {
	i := splitExpandIndex(ce)
	n := int(t.expandElem[i])
	i++
	for _, ce := range t.expandElem[i : i+n] {
		w = append(w, Elem(ce))
	}
	return w
}

func (t *table) matchContraction(w []Elem, ce Elem, suffix []byte) ([]Elem, int) {
	index, n, offset := splitContractIndex(ce)

	scan := t.contractTries.scanner(index, n, suffix)
	buf := [norm.MaxSegmentSize]byte{}
	bufp := 0
	p := scan.scan(0)

	if !scan.done && p < len(suffix) && suffix[p] >= utf8.RuneSelf {
		// By now we should have filtered most cases.
		p0 := p
		bufn := 0
		rune := norm.NFD.Properties(suffix[p:])
		p += rune.Size()
		if rune.LeadCCC() != 0 {
			prevCC := rune.TrailCCC()
			// A gap may only occur in the last normalization segment.
			// This also ensures that len(scan.s) < norm.MaxSegmentSize.
			if end := norm.NFD.FirstBoundary(suffix[p:]); end != -1 {
				scan.s = suffix[:p+end]
			}
			for p < len(suffix) && !scan.done && suffix[p] >= utf8.RuneSelf {
				rune = norm.NFD.Properties(suffix[p:])
				if ccc := rune.LeadCCC(); ccc == 0 || prevCC >= ccc {
					break
				}
				prevCC = rune.TrailCCC()
				if pp := scan.scan(p); pp != p {
					// Copy the interstitial runes for later processing.
					bufn += copy(buf[bufn:], suffix[p0:p])
					if scan.pindex == pp {
						bufp = bufn
					}
					p, p0 = pp, pp
				} else {
					p += rune.Size()
				}
			}
		}
	}
	// Append weights for the matched contraction, which may be an expansion.
	i, n := scan.result()
	ce = Elem(t.contractElem[i+offset])
	if ce.ctype() == ceNormal {
		w = append(w, ce)
	} else {
		w = t.appendExpansion(w, ce)
	}
	// Append weights for the runes in the segment not part of the contraction.
	for b, p := buf[:bufp], 0; len(b) > 0; b = b[p:] {
		w, p = t.appendNext(w, source{bytes: b})
	}
	return w, n
}

// TODO: unify the two implementations. This is best done after first simplifying
// the algorithm taking into account the inclusion of both NFC and NFD forms
// in the table.
func (t *table) matchContractionString(w []Elem, ce Elem, suffix string) ([]Elem, int) {
	index, n, offset := splitContractIndex(ce)

	scan := t.contractTries.scannerString(index, n, suffix)
	buf := [norm.MaxSegmentSize]byte{}
	bufp := 0
	p := scan.scan(0)

	if !scan.done && p < len(suffix) && suffix[p] >= utf8.RuneSelf {
		// By now we should have filtered most cases.
		p0 := p
		bufn := 0
		rune := norm.NFD.PropertiesString(suffix[p:])
		p += rune.Size()
		if rune.LeadCCC() != 0 {
			prevCC := rune.TrailCCC()
			// A gap may only occur in the last normalization segment.
			// This also ensures that len(scan.s) < norm.MaxSegmentSize.
			if end := norm.NFD.FirstBoundaryInString(suffix[p:]); end != -1 {
				scan.s = suffix[:p+end]
			}
			for p < len(suffix) && !scan.done && suffix[p] >= utf8.RuneSelf {
				rune = norm.NFD.PropertiesString(suffix[p:])
				if ccc := rune.LeadCCC(); ccc == 0 || prevCC >= ccc {
					break
				}
				prevCC = rune.TrailCCC()
				if pp := scan.scan(p); pp != p {
					// Copy the interstitial runes for later processing.
					bufn += copy(buf[bufn:], suffix[p0:p])
					if scan.pindex == pp {
						bufp = bufn
					}
					p, p0 = pp, pp
				} else {
					p += rune.Size()
				}
			}
		}
	}
	// Append weights for the matched contraction, which may be an expansion.
	i, n := scan.result()
	ce = Elem(t.contractElem[i+offset])
	if ce.ctype() == ceNormal {
		w = append(w, ce)
	} else {
		w = t.appendExpansion(w, ce)
	}
	// Append weights for the runes in the segment not part of the contraction.
	for b, p := buf[:bufp], 0; len(b) > 0; b = b[p:] {
		w, p = t.appendNext(w, source{bytes: b})
	}
	return w, n
}

// TODO: this should stay after the rest of this file is moved to colltab
func (t tableIndex) TrieIndex() []uint16 {
	return mainLookup[:]
}

func (t tableIndex) TrieValues() []uint32 {
	return mainValues[:]
}

func (t tableIndex) FirstBlockOffsets() (lookup, value uint16) {
	return uint16(t.lookupOffset), uint16(t.valuesOffset)
}

func (t tableIndex) ExpandElems() []uint32 {
	return mainExpandElem[:]
}

func (t tableIndex) ContractTries() []struct{ l, h, n, i uint8 } {
	return mainCTEntries[:]
}

func (t tableIndex) ContractElems() []uint32 {
	return mainContractElem[:]
}

func (t tableIndex) MaxContractLen() int {
	return 18
}

func (t tableIndex) VariableTop() uint32 {
	return 0x30E
}
