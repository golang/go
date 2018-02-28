// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

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

	for w&1 == 0 {
		w >>= 1
		i++
	}

	return i
}

func (bv bvec) IsEmpty() bool {
	for i := int32(0); i < bv.n; i += wordBits {
		if bv.b[i>>wordShift] != 0 {
			return false
		}
	}
	return true
}

func (bv bvec) Not() {
	i := int32(0)
	w := int32(0)
	for ; i < bv.n; i, w = i+wordBits, w+1 {
		bv.b[w] = ^bv.b[w]
	}
}

// union
func (dst bvec) Or(src1, src2 bvec) {
	for i, x := range src1.b {
		dst.b[i] = x | src2.b[i]
	}
}

// intersection
func (dst bvec) And(src1, src2 bvec) {
	for i, x := range src1.b {
		dst.b[i] = x & src2.b[i]
	}
}

// difference
func (dst bvec) AndNot(src1, src2 bvec) {
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
