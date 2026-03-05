// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flate

import (
	"math/bits"
)

const (
	// tableBits is the number of bits used in the hash table.
	tableBits = 15

	// tableSize is the size of the hash table.
	tableSize = 1 << tableBits

	// hashLongBytes is the number of bytes used for long table hashes.
	hashLongBytes = 7

	// baseMatchOffset is the smallest match offset.
	baseMatchOffset = 1

	// baseMatchLength is the smallest match length per RFC section 3.2.5.
	baseMatchLength = 3

	// maxMatchOffset is the largest match offset.
	maxMatchOffset = 1 << 15

	// allocHistory is the size to preallocate for history.
	allocHistory = maxStoreBlockSize * 5

	// bufferReset is the buffer offset at which the history is reset.
	bufferReset = (1 << 31) - allocHistory - maxStoreBlockSize - 1
)

// fastEncL1 to fastEncL6 provides specialized encoders for levels 1-6
// that each provide a different speed/size/memory strategies.
//
// Level 1: Single small table, 5 byte hashes, sparse indexing.
// Level 2: Single big table, 5 byte hashes, indexing ~ every 2 bytes.
// Level 3: Single medium table, 5 byte hashes, 2 candidates per table entry.
// Level 4: Two tables, 4/7 byte hashes, 1 candidate per table entry.
// Level 5: Two tables, 4/7 byte hashes, 2 candidates per 7-byte table entry.
// Level 6: Two tables, 4/7 byte hashes, full indexing, checks for repeats.
//
// Skipping on contiguous non-matches also decreases as levels go up.

// fastEnc is the interface implemented by the level 1-6 fast encoders.
type fastEnc interface {
	// encode src into dst.
	encode(dst *tokens, src []byte)
	// reset the encoder so matches are not made with previous data.
	reset()
}

// newFastEnc returns a fastEnc encoder for the given compression level (1-6).
func newFastEnc(level int) fastEnc {
	switch level {
	case 1:
		return &fastEncL1{fastGen: fastGen{cur: maxStoreBlockSize}}
	case 2:
		return &fastEncL2{fastGen: fastGen{cur: maxStoreBlockSize}}
	case 3:
		return &fastEncL3{fastGen: fastGen{cur: maxStoreBlockSize}}
	case 4:
		return &fastEncL4{fastGen: fastGen{cur: maxStoreBlockSize}}
	case 5:
		return &fastEncL5{fastGen: fastGen{cur: maxStoreBlockSize}}
	case 6:
		return &fastEncL6{fastGen: fastGen{cur: maxStoreBlockSize}}
	default:
		panic("invalid level specified")
	}
}

// fastGen maintains the table for matches,
// and the previous byte block for level 1 and up.
// This is the generic implementation.
type fastGen struct {
	hist []byte
	cur  int32
}

// addBlock appends src to the history and returns the offset where src starts in e.hist.
func (e *fastGen) addBlock(src []byte) int32 {
	// check if we have space already
	if len(e.hist)+len(src) > cap(e.hist) {
		if cap(e.hist) == 0 {
			e.hist = make([]byte, 0, allocHistory)
		} else {
			if cap(e.hist) < maxMatchOffset*2 {
				panic("unexpected buffer size")
			}
			// Move down
			offset := int32(len(e.hist)) - maxMatchOffset
			copy(e.hist[0:maxMatchOffset], e.hist[offset:offset+maxMatchOffset])
			e.cur += offset
			e.hist = e.hist[:maxMatchOffset]
		}
	}
	s := int32(len(e.hist))
	e.hist = append(e.hist, src...)
	return s
}

// matchLenLimited returns the match length between offsets s and t in src.
// The maximum length returned is maxMatchLength - 4.
// It is assumed that s > t, that t >= 0 and s < len(src).
func (e *fastGen) matchLenLimited(s, t int, src []byte) int32 {
	a := src[s:min(s+maxMatchLength-4, len(src))]
	b := src[t:]
	return int32(matchLen(a, b))
}

// matchLenLong returns the match length between offsets s and t in src.
// It is assumed that s > t, that t >= 0 and s < len(src).
func (e *fastGen) matchLenLong(s, t int, src []byte) int32 {
	return int32(matchLen(src[s:], src[t:]))
}

// reset resets the encoding table to prepare for a new compression stream.
func (e *fastGen) reset() {
	if cap(e.hist) < allocHistory {
		e.hist = make([]byte, 0, allocHistory)
	}
	// We offset current position so everything will be out of reach.
	// If we are above the buffer reset it will be cleared anyway since len(hist) == 0.
	if e.cur <= bufferReset {
		e.cur += maxMatchOffset + int32(len(e.hist))
	}
	e.hist = e.hist[:0]
}

func (f *fastGen) getFastGen() *fastGen { return f }

// tableEntry stores the offset of a hash match in the input history.
type tableEntry struct {
	offset int32
}

// tableEntryPrev stores the current and previous offsets for a hash entry.
type tableEntryPrev struct {
	cur  tableEntry
	prev tableEntry
}

const (
	prime3bytes = 506832829
	prime4bytes = 2654435761
	prime5bytes = 889523592379
	prime6bytes = 227718039650203
	prime7bytes = 58295818150454627
	prime8bytes = 0xcf1bbcdcb7a56463
)

// hashLen returns a hash of the first n bytes of u, using b output bits.
// It expects 3 <= n <= 8; other values are treated as n == 4.
// The bit length b must be <= 32.
// b and n should be constants in speed-critical use.
func hashLen(u uint64, b, n uint8) uint32 {
	switch n {
	case 3:
		return (uint32(u<<8) * prime3bytes) >> (32 - b)
	case 5:
		return uint32(((u << (64 - 40)) * prime5bytes) >> (64 - b))
	case 6:
		return uint32(((u << (64 - 48)) * prime6bytes) >> (64 - b))
	case 7:
		return uint32(((u << (64 - 56)) * prime7bytes) >> (64 - b))
	case 8:
		return uint32((u * prime8bytes) >> (64 - b))
	default:
		return (uint32(u) * prime4bytes) >> (32 - b)
	}
}

// matchLen returns the maximum common prefix length of a and b.
// a must be the shortest of the two.
func matchLen(a, b []byte) (n int) {
	left := len(a)
	for left >= 8 {
		diff := loadLE64(a, n) ^ loadLE64(b, n)
		if diff != 0 {
			return n + bits.TrailingZeros64(diff)>>3
		}
		n += 8
		left -= 8
	}

	a = a[n:]
	b = b[n:]
	b = b[:len(a)]
	for i := range a {
		if a[i] != b[i] {
			break
		}
		n++
	}
	return n
}
