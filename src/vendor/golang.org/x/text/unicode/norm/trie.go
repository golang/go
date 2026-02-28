// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package norm

type valueRange struct {
	value  uint16 // header: value:stride
	lo, hi byte   // header: lo:n
}

type sparseBlocks struct {
	values []valueRange
	offset []uint16
}

var nfcSparse = sparseBlocks{
	values: nfcSparseValues[:],
	offset: nfcSparseOffset[:],
}

var nfkcSparse = sparseBlocks{
	values: nfkcSparseValues[:],
	offset: nfkcSparseOffset[:],
}

var (
	nfcData  = newNfcTrie(0)
	nfkcData = newNfkcTrie(0)
)

// lookup determines the type of block n and looks up the value for b.
// For n < t.cutoff, the block is a simple lookup table. Otherwise, the block
// is a list of ranges with an accompanying value. Given a matching range r,
// the value for b is by r.value + (b - r.lo) * stride.
func (t *sparseBlocks) lookup(n uint32, b byte) uint16 {
	offset := t.offset[n]
	header := t.values[offset]
	lo := offset + 1
	hi := lo + uint16(header.lo)
	for lo < hi {
		m := lo + (hi-lo)/2
		r := t.values[m]
		if r.lo <= b && b <= r.hi {
			return r.value + uint16(b-r.lo)*header.value
		}
		if b < r.lo {
			hi = m
		} else {
			lo = m + 1
		}
	}
	return 0
}
