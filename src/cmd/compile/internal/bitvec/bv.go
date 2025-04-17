// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bitvec

import (
	"math/bits"

	"cmd/compile/internal/base"
	"cmd/internal/src"
)

const (
	wordBits  = 32
	wordMask  = wordBits - 1
	wordShift = 5
)

// A BitVec is a bit vector.
type BitVec struct {
	N int32    // number of bits in vector
	B []uint32 // words holding bits
}

func New(n int32) BitVec {
	nword := (n + wordBits - 1) / wordBits
	return BitVec{n, make([]uint32, nword)}
}

type Bulk struct {
	words []uint32
	nbit  int32
	nword int32
}

func NewBulk(nbit int32, count int32, pos src.XPos) Bulk {
	nword := (nbit + wordBits - 1) / wordBits
	size := int64(nword) * int64(count)
	if int64(int32(size*4)) != size*4 {
		base.FatalfAt(pos, "NewBulk too big: nbit=%d count=%d nword=%d size=%d", nbit, count, nword, size)
	}
	return Bulk{
		words: make([]uint32, size),
		nbit:  nbit,
		nword: nword,
	}
}

func (b *Bulk) Next() BitVec {
	out := BitVec{b.nbit, b.words[:b.nword]}
	b.words = b.words[b.nword:]
	return out
}

func (bv1 BitVec) Eq(bv2 BitVec) bool {
	if bv1.N != bv2.N {
		base.Fatalf("bvequal: lengths %d and %d are not equal", bv1.N, bv2.N)
	}
	for i, x := range bv1.B {
		if x != bv2.B[i] {
			return false
		}
	}
	return true
}

func (dst BitVec) Copy(src BitVec) {
	copy(dst.B, src.B)
}

func (bv BitVec) Get(i int32) bool {
	if i < 0 || i >= bv.N {
		base.Fatalf("bvget: index %d is out of bounds with length %d\n", i, bv.N)
	}
	mask := uint32(1 << uint(i%wordBits))
	return bv.B[i>>wordShift]&mask != 0
}

func (bv BitVec) Set(i int32) {
	if i < 0 || i >= bv.N {
		base.Fatalf("bvset: index %d is out of bounds with length %d\n", i, bv.N)
	}
	mask := uint32(1 << uint(i%wordBits))
	bv.B[i/wordBits] |= mask
}

func (bv BitVec) Unset(i int32) {
	if i < 0 || i >= bv.N {
		base.Fatalf("bvunset: index %d is out of bounds with length %d\n", i, bv.N)
	}
	mask := uint32(1 << uint(i%wordBits))
	bv.B[i/wordBits] &^= mask
}

// bvnext returns the smallest index >= i for which bvget(bv, i) == 1.
// If there is no such index, bvnext returns -1.
func (bv BitVec) Next(i int32) int32 {
	if i >= bv.N {
		return -1
	}

	// Jump i ahead to next word with bits.
	if bv.B[i>>wordShift]>>uint(i&wordMask) == 0 {
		i &^= wordMask
		i += wordBits
		for i < bv.N && bv.B[i>>wordShift] == 0 {
			i += wordBits
		}
	}

	if i >= bv.N {
		return -1
	}

	// Find 1 bit.
	w := bv.B[i>>wordShift] >> uint(i&wordMask)
	i += int32(bits.TrailingZeros32(w))

	return i
}

func (bv BitVec) IsEmpty() bool {
	for _, x := range bv.B {
		if x != 0 {
			return false
		}
	}
	return true
}

func (bv BitVec) Count() int {
	n := 0
	for _, x := range bv.B {
		n += bits.OnesCount32(x)
	}
	return n
}

func (bv BitVec) Not() {
	for i, x := range bv.B {
		bv.B[i] = ^x
	}
	if bv.N%wordBits != 0 {
		bv.B[len(bv.B)-1] &= 1<<uint(bv.N%wordBits) - 1 // clear bits past N in the last word
	}
}

// union
func (dst BitVec) Or(src1, src2 BitVec) {
	if len(src1.B) == 0 {
		return
	}
	_, _ = dst.B[len(src1.B)-1], src2.B[len(src1.B)-1] // hoist bounds checks out of the loop

	for i, x := range src1.B {
		dst.B[i] = x | src2.B[i]
	}
}

// intersection
func (dst BitVec) And(src1, src2 BitVec) {
	if len(src1.B) == 0 {
		return
	}
	_, _ = dst.B[len(src1.B)-1], src2.B[len(src1.B)-1] // hoist bounds checks out of the loop

	for i, x := range src1.B {
		dst.B[i] = x & src2.B[i]
	}
}

// difference
func (dst BitVec) AndNot(src1, src2 BitVec) {
	if len(src1.B) == 0 {
		return
	}
	_, _ = dst.B[len(src1.B)-1], src2.B[len(src1.B)-1] // hoist bounds checks out of the loop

	for i, x := range src1.B {
		dst.B[i] = x &^ src2.B[i]
	}
}

func (bv BitVec) String() string {
	s := make([]byte, 2+bv.N)
	copy(s, "#*")
	for i := int32(0); i < bv.N; i++ {
		ch := byte('0')
		if bv.Get(i) {
			ch = '1'
		}
		s[2+i] = ch
	}
	return string(s)
}

func (bv BitVec) Clear() {
	clear(bv.B)
}
