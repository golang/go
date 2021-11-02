// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package liveness

import "cmd/compile/internal/bitvec"

// FNV-1 hash function constants.
const (
	h0 = 2166136261
	hp = 16777619
)

// bvecSet is a set of bvecs, in initial insertion order.
type bvecSet struct {
	index []int           // hash -> uniq index. -1 indicates empty slot.
	uniq  []bitvec.BitVec // unique bvecs, in insertion order
}

func (m *bvecSet) grow() {
	// Allocate new index.
	n := len(m.index) * 2
	if n == 0 {
		n = 32
	}
	newIndex := make([]int, n)
	for i := range newIndex {
		newIndex[i] = -1
	}

	// Rehash into newIndex.
	for i, bv := range m.uniq {
		h := hashbitmap(h0, bv) % uint32(len(newIndex))
		for {
			j := newIndex[h]
			if j < 0 {
				newIndex[h] = i
				break
			}
			h++
			if h == uint32(len(newIndex)) {
				h = 0
			}
		}
	}
	m.index = newIndex
}

// add adds bv to the set and returns its index in m.extractUnique,
// and whether it is newly added.
// If it is newly added, the caller must not modify bv after this.
func (m *bvecSet) add(bv bitvec.BitVec) (int, bool) {
	if len(m.uniq)*4 >= len(m.index) {
		m.grow()
	}

	index := m.index
	h := hashbitmap(h0, bv) % uint32(len(index))
	for {
		j := index[h]
		if j < 0 {
			// New bvec.
			index[h] = len(m.uniq)
			m.uniq = append(m.uniq, bv)
			return len(m.uniq) - 1, true
		}
		jlive := m.uniq[j]
		if bv.Eq(jlive) {
			// Existing bvec.
			return j, false
		}

		h++
		if h == uint32(len(index)) {
			h = 0
		}
	}
}

// extractUnique returns this slice of unique bit vectors in m, as
// indexed by the result of bvecSet.add.
func (m *bvecSet) extractUnique() []bitvec.BitVec {
	return m.uniq
}

func hashbitmap(h uint32, bv bitvec.BitVec) uint32 {
	n := int((bv.N + 31) / 32)
	for i := 0; i < n; i++ {
		w := bv.B[i]
		h = (h * hp) ^ (w & 0xff)
		h = (h * hp) ^ ((w >> 8) & 0xff)
		h = (h * hp) ^ ((w >> 16) & 0xff)
		h = (h * hp) ^ ((w >> 24) & 0xff)
	}

	return h
}
