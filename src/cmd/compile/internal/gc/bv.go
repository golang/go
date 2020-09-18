// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"math/bits"
)

const (
	wordBits  = 32
	wordMask  = wordBits - 1
	wordShift = 5
)

// A bvec is a bit vector.
type bvec struct {
	n int32    // number of bits in vector
	b []uint32 // words holding bits
}

func bvalloc(n int32) bvec {
	nword := (n + wordBits - 1) / wordBits
	return bvec{n, make([]uint32, nword)}
}

type bulkBvec struct {
	words []uint32
	nbit  int32
	nword int32
}

func bvbulkalloc(nbit int32, count int32) bulkBvec {
	nword := (nbit + wordBits - 1) / wordBits
	size := int64(nword) * int64(count)
	if int64(int32(size*4)) != size*4 {
		Fatalf("bvbulkalloc too big: nbit=%d count=%d nword=%d size=%d", nbit, count, nword, size)
	}
	return bulkBvec{
		words: make([]uint32, size),
		nbit:  nbit,
		nword: nword,
	}
}

func (b *bulkBvec) next() bvec {
	out := bvec{b.nbit, b.words[:b.nword]}
	b.words = b.words[b.nword:]
	return out
}

func (bv1 bvec) Eq(bv2 bvec) bool {
	if bv1.n != bv2.n {
		Fatalf("bvequal: lengths %d and %d are not equal", bv1.n, bv2.n)
	}
	for i, x := range bv1.b {
		if x != bv2.b[i] {
			return false
		}
	}
	return true
}

func (dst bvec) Copy(src bvec) {
	copy(dst.b, src.b)
}

func (bv bvec) Get(i int32) bool {
	if i < 0 || i >= bv.n {
		Fatalf("bvget: index %d is out of bounds with length %d\n", i, bv.n)
	}
	mask := uint32(1 << uint(i%wordBits))
	return bv.b[i>>wordShift]&mask != 0
}

func (bv bvec) Set(i int32) {
	if i < 0 || i >= bv.n {
		Fatalf("bvset: index %d is out of bounds with length %d\n", i, bv.n)
	}
	mask := uint32(1 << uint(i%wordBits))
	bv.b[i/wordBits] |= mask
}

func (bv bvec) Unset(i int32) {
	if i < 0 || i >= bv.n {
		Fatalf("bvunset: index %d is out of bounds with length %d\n", i, bv.n)
	}
	mask := uint32(1 << uint(i%wordBits))
	bv.b[i/wordBits] &^= mask
}

// bvnext returns the smallest index >= i for which bvget(bv, i) == 1.
// If there is no such index, bvnext returns -1.
func (bv bvec) Next(i int32) int32 {
	if i >= bv.n {
		return -1
	}

	// Jump i ahead to next word with bits.
	if bv.b[i>>wordShift]>>uint(i&wordMask) == 0 {
		i &^= wordMask
		i += wordBits
		for i < bv.n && bv.b[i>>wordShift] == 0 {
			i += wordBits
		}
	}

	if i >= bv.n {
		return -1
	}

	// Find 1 bit.
	w := bv.b[i>>wordShift] >> uint(i&wordMask)
	i += int32(bits.TrailingZeros32(w))

	return i
}

func (bv bvec) IsEmpty() bool {
	for _, x := range bv.b {
		if x != 0 {
			return false
		}
	}
	return true
}

func (bv bvec) Not() {
	for i, x := range bv.b {
		bv.b[i] = ^x
	}
}

// union
func (dst bvec) Or(src1, src2 bvec) {
	if len(src1.b) == 0 {
		return
	}
	_, _ = dst.b[len(src1.b)-1], src2.b[len(src1.b)-1] // hoist bounds checks out of the loop

	for i, x := range src1.b {
		dst.b[i] = x | src2.b[i]
	}
}

// intersection
func (dst bvec) And(src1, src2 bvec) {
	if len(src1.b) == 0 {
		return
	}
	_, _ = dst.b[len(src1.b)-1], src2.b[len(src1.b)-1] // hoist bounds checks out of the loop

	for i, x := range src1.b {
		dst.b[i] = x & src2.b[i]
	}
}

// difference
func (dst bvec) AndNot(src1, src2 bvec) {
	if len(src1.b) == 0 {
		return
	}
	_, _ = dst.b[len(src1.b)-1], src2.b[len(src1.b)-1] // hoist bounds checks out of the loop

	for i, x := range src1.b {
		dst.b[i] = x &^ src2.b[i]
	}
}

func (bv bvec) String() string {
	s := make([]byte, 2+bv.n)
	copy(s, "#*")
	for i := int32(0); i < bv.n; i++ {
		ch := byte('0')
		if bv.Get(i) {
			ch = '1'
		}
		s[2+i] = ch
	}
	return string(s)
}

func (bv bvec) Clear() {
	for i := range bv.b {
		bv.b[i] = 0
	}
}

// FNV-1 hash function constants.
const (
	H0 = 2166136261
	Hp = 16777619
)

func hashbitmap(h uint32, bv bvec) uint32 {
	n := int((bv.n + 31) / 32)
	for i := 0; i < n; i++ {
		w := bv.b[i]
		h = (h * Hp) ^ (w & 0xff)
		h = (h * Hp) ^ ((w >> 8) & 0xff)
		h = (h * Hp) ^ ((w >> 16) & 0xff)
		h = (h * Hp) ^ ((w >> 24) & 0xff)
	}

	return h
}

// bvecSet is a set of bvecs, in initial insertion order.
type bvecSet struct {
	index []int  // hash -> uniq index. -1 indicates empty slot.
	uniq  []bvec // unique bvecs, in insertion order
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
		h := hashbitmap(H0, bv) % uint32(len(newIndex))
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

// add adds bv to the set and returns its index in m.extractUniqe.
// The caller must not modify bv after this.
func (m *bvecSet) add(bv bvec) int {
	if len(m.uniq)*4 >= len(m.index) {
		m.grow()
	}

	index := m.index
	h := hashbitmap(H0, bv) % uint32(len(index))
	for {
		j := index[h]
		if j < 0 {
			// New bvec.
			index[h] = len(m.uniq)
			m.uniq = append(m.uniq, bv)
			return len(m.uniq) - 1
		}
		jlive := m.uniq[j]
		if bv.Eq(jlive) {
			// Existing bvec.
			return j
		}

		h++
		if h == uint32(len(index)) {
			h = 0
		}
	}
}

// extractUniqe returns this slice of unique bit vectors in m, as
// indexed by the result of bvecSet.add.
func (m *bvecSet) extractUniqe() []bvec {
	return m.uniq
}
