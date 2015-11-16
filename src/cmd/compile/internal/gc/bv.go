// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import "fmt"

const (
	WORDSIZE  = 4
	WORDBITS  = 32
	WORDMASK  = WORDBITS - 1
	WORDSHIFT = 5
)

// A Bvec is a bit vector.
type Bvec struct {
	n int32    // number of bits in vector
	b []uint32 // words holding bits
}

func bvsize(n uint32) uint32 {
	return ((n + WORDBITS - 1) / WORDBITS) * WORDSIZE
}

func bvbits(bv Bvec) int32 {
	return bv.n
}

func bvwords(bv Bvec) int32 {
	return (bv.n + WORDBITS - 1) / WORDBITS
}

func bvalloc(n int32) Bvec {
	return Bvec{n, make([]uint32, bvsize(uint32(n))/4)}
}

type bulkBvec struct {
	words []uint32
	nbit  int32
	nword int32
}

func bvbulkalloc(nbit int32, count int32) bulkBvec {
	nword := (nbit + WORDBITS - 1) / WORDBITS
	return bulkBvec{
		words: make([]uint32, nword*count),
		nbit:  nbit,
		nword: nword,
	}
}

func (b *bulkBvec) next() Bvec {
	out := Bvec{b.nbit, b.words[:b.nword]}
	b.words = b.words[b.nword:]
	return out
}

// difference
func bvandnot(dst Bvec, src1 Bvec, src2 Bvec) {
	for i, x := range src1.b {
		dst.b[i] = x &^ src2.b[i]
	}
}

func bvcmp(bv1 Bvec, bv2 Bvec) int {
	if bv1.n != bv2.n {
		Fatalf("bvequal: lengths %d and %d are not equal", bv1.n, bv2.n)
	}
	for i, x := range bv1.b {
		if x != bv2.b[i] {
			return 1
		}
	}
	return 0
}

func bvcopy(dst Bvec, src Bvec) {
	for i, x := range src.b {
		dst.b[i] = x
	}
}

func bvconcat(src1 Bvec, src2 Bvec) Bvec {
	dst := bvalloc(src1.n + src2.n)
	for i := int32(0); i < src1.n; i++ {
		if bvget(src1, i) != 0 {
			bvset(dst, i)
		}
	}
	for i := int32(0); i < src2.n; i++ {
		if bvget(src2, i) != 0 {
			bvset(dst, i+src1.n)
		}
	}
	return dst
}

func bvget(bv Bvec, i int32) int {
	if i < 0 || i >= bv.n {
		Fatalf("bvget: index %d is out of bounds with length %d\n", i, bv.n)
	}
	return int((bv.b[i>>WORDSHIFT] >> uint(i&WORDMASK)) & 1)
}

// bvnext returns the smallest index >= i for which bvget(bv, i) == 1.
// If there is no such index, bvnext returns -1.
func bvnext(bv Bvec, i int32) int {
	if i >= bv.n {
		return -1
	}

	// Jump i ahead to next word with bits.
	if bv.b[i>>WORDSHIFT]>>uint(i&WORDMASK) == 0 {
		i &^= WORDMASK
		i += WORDBITS
		for i < bv.n && bv.b[i>>WORDSHIFT] == 0 {
			i += WORDBITS
		}
	}

	if i >= bv.n {
		return -1
	}

	// Find 1 bit.
	w := bv.b[i>>WORDSHIFT] >> uint(i&WORDMASK)

	for w&1 == 0 {
		w >>= 1
		i++
	}

	return int(i)
}

func bvisempty(bv Bvec) bool {
	for i := int32(0); i < bv.n; i += WORDBITS {
		if bv.b[i>>WORDSHIFT] != 0 {
			return false
		}
	}
	return true
}

func bvnot(bv Bvec) {
	i := int32(0)
	w := int32(0)
	for ; i < bv.n; i, w = i+WORDBITS, w+1 {
		bv.b[w] = ^bv.b[w]
	}
}

// union
func bvor(dst Bvec, src1 Bvec, src2 Bvec) {
	for i, x := range src1.b {
		dst.b[i] = x | src2.b[i]
	}
}

// intersection
func bvand(dst Bvec, src1 Bvec, src2 Bvec) {
	for i, x := range src1.b {
		dst.b[i] = x & src2.b[i]
	}
}

func bvprint(bv Bvec) {
	fmt.Printf("#*")
	for i := int32(0); i < bv.n; i++ {
		fmt.Printf("%d", bvget(bv, i))
	}
}

func bvreset(bv Bvec, i int32) {
	if i < 0 || i >= bv.n {
		Fatalf("bvreset: index %d is out of bounds with length %d\n", i, bv.n)
	}
	mask := uint32(^(1 << uint(i%WORDBITS)))
	bv.b[i/WORDBITS] &= mask
}

func bvresetall(bv Bvec) {
	for i := range bv.b {
		bv.b[i] = 0
	}
}

func bvset(bv Bvec, i int32) {
	if i < 0 || i >= bv.n {
		Fatalf("bvset: index %d is out of bounds with length %d\n", i, bv.n)
	}
	mask := uint32(1 << uint(i%WORDBITS))
	bv.b[i/WORDBITS] |= mask
}
