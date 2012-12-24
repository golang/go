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

// appendNext appends the weights corresponding to the next rune or
// contraction in s.  If a contraction is matched to a discontinuous
// sequence of runes, the weights for the interstitial runes are
// appended as well.  It returns a new slice that includes the appended
// weights and the number of bytes consumed from s.
func (t *table) appendNext(w []colElem, src source) (res []colElem, n int) {
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
				for b := src.nfd(hangulSize); len(b) > 0; b = b[sz:] {
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
		src = src.tail(sz)
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

func (t *table) appendExpansion(w []colElem, ce colElem) []colElem {
	i := splitExpandIndex(ce)
	n := int(t.expandElem[i])
	i++
	for _, ce := range t.expandElem[i : i+n] {
		w = append(w, colElem(ce))
	}
	return w
}

func (t *table) matchContraction(w []colElem, ce colElem, suffix []byte) ([]colElem, int) {
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
	ce = colElem(t.contractElem[i+offset])
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
func (t *table) matchContractionString(w []colElem, ce colElem, suffix string) ([]colElem, int) {
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
	ce = colElem(t.contractElem[i+offset])
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
