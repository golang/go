// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flate

// This encoding algorithm, which prioritizes speed over output size, is
// based on Snappy's LZ77-style encoder: github.com/golang/snappy

const (
	tableBits  = 14             // Bits used in the table.
	tableSize  = 1 << tableBits // Size of the table.
	tableMask  = tableSize - 1  // Mask for table indices. Redundant, but can eliminate bounds checks.
	tableShift = 32 - tableBits // Right-shift to get the tableBits most significant bits of a uint32.
)

func load32(b []byte, i int32) uint32 {
	b = b[i : i+4 : len(b)] // Help the compiler eliminate bounds checks on the next line.
	return uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24
}

func load64(b []byte, i int32) uint64 {
	b = b[i : i+8 : len(b)] // Help the compiler eliminate bounds checks on the next line.
	return uint64(b[0]) | uint64(b[1])<<8 | uint64(b[2])<<16 | uint64(b[3])<<24 |
		uint64(b[4])<<32 | uint64(b[5])<<40 | uint64(b[6])<<48 | uint64(b[7])<<56
}

func hash(u uint32) uint32 {
	return (u * 0x1e35a7bd) >> tableShift
}

// These constants are defined by the Snappy implementation so that its
// assembly implementation can fast-path some 16-bytes-at-a-time copies. They
// aren't necessary in the pure Go implementation, as we don't use those same
// optimizations, but using the same thresholds doesn't really hurt.
const (
	inputMargin            = 16 - 1
	minNonLiteralBlockSize = 1 + 1 + inputMargin
)

type tableEntry struct {
	val    uint32 // Value at destination
	offset int32
}

// deflateFast maintains the table for matches,
// and the previous byte block for cross block matching.
type deflateFast struct {
	table [tableSize]tableEntry
	prev  []byte // Previous block, zero length if unknown.
	cur   int32  // Current match offset.
}

func newDeflateFast() *deflateFast {
	return &deflateFast{cur: maxStoreBlockSize, prev: make([]byte, 0, maxStoreBlockSize)}
}

// encode encodes a block given in src and appends tokens
// to dst and returns the result.
func (e *deflateFast) encode(dst []token, src []byte) []token {
	// Ensure that e.cur doesn't wrap.
	if e.cur > 1<<30 {
		e.resetAll()
	}

	// This check isn't in the Snappy implementation, but there, the caller
	// instead of the callee handles this case.
	if len(src) < minNonLiteralBlockSize {
		e.cur += maxStoreBlockSize
		e.prev = e.prev[:0]
		return emitLiteral(dst, src)
	}

	// sLimit is when to stop looking for offset/length copies. The inputMargin
	// lets us use a fast path for emitLiteral in the main loop, while we are
	// looking for copies.
	sLimit := int32(len(src) - inputMargin)

	// nextEmit is where in src the next emitLiteral should start from.
	nextEmit := int32(0)
	s := int32(0)
	cv := load32(src, s)
	nextHash := hash(cv)

	for {
		// Copied from the C++ snappy implementation:
		//
		// Heuristic match skipping: If 32 bytes are scanned with no matches
		// found, start looking only at every other byte. If 32 more bytes are
		// scanned (or skipped), look at every third byte, etc.. When a match
		// is found, immediately go back to looking at every byte. This is a
		// small loss (~5% performance, ~0.1% density) for compressible data
		// due to more bookkeeping, but for non-compressible data (such as
		// JPEG) it's a huge win since the compressor quickly "realizes" the
		// data is incompressible and doesn't bother looking for matches
		// everywhere.
		//
		// The "skip" variable keeps track of how many bytes there are since
		// the last match; dividing it by 32 (ie. right-shifting by five) gives
		// the number of bytes to move ahead for each iteration.
		skip := int32(32)

		nextS := s
		var candidate tableEntry
		for {
			s = nextS
			bytesBetweenHashLookups := skip >> 5
			nextS = s + bytesBetweenHashLookups
			skip += bytesBetweenHashLookups
			if nextS > sLimit {
				goto emitRemainder
			}
			candidate = e.table[nextHash&tableMask]
			now := load32(src, nextS)
			e.table[nextHash&tableMask] = tableEntry{offset: s + e.cur, val: cv}
			nextHash = hash(now)

			offset := s - (candidate.offset - e.cur)
			if offset > maxMatchOffset || cv != candidate.val {
				// Out of range or not matched.
				cv = now
				continue
			}
			break
		}

		// A 4-byte match has been found. We'll later see if more than 4 bytes
		// match. But, prior to the match, src[nextEmit:s] are unmatched. Emit
		// them as literal bytes.
		dst = emitLiteral(dst, src[nextEmit:s])

		// Call emitCopy, and then see if another emitCopy could be our next
		// move. Repeat until we find no match for the input immediately after
		// what was consumed by the last emitCopy call.
		//
		// If we exit this loop normally then we need to call emitLiteral next,
		// though we don't yet know how big the literal will be. We handle that
		// by proceeding to the next iteration of the main loop. We also can
		// exit this loop via goto if we get close to exhausting the input.
		for {
			// Invariant: we have a 4-byte match at s, and no need to emit any
			// literal bytes prior to s.

			// Extend the 4-byte match as long as possible.
			//
			s += 4
			t := candidate.offset - e.cur + 4
			l := e.matchLen(s, t, src)

			// matchToken is flate's equivalent of Snappy's emitCopy. (length,offset)
			dst = append(dst, matchToken(uint32(l+4-baseMatchLength), uint32(s-t-baseMatchOffset)))
			s += l
			nextEmit = s
			if s >= sLimit {
				goto emitRemainder
			}

			// We could immediately start working at s now, but to improve
			// compression we first update the hash table at s-1 and at s. If
			// another emitCopy is not our next move, also calculate nextHash
			// at s+1. At least on GOARCH=amd64, these three hash calculations
			// are faster as one load64 call (with some shifts) instead of
			// three load32 calls.
			x := load64(src, s-1)
			prevHash := hash(uint32(x))
			e.table[prevHash&tableMask] = tableEntry{offset: e.cur + s - 1, val: uint32(x)}
			x >>= 8
			currHash := hash(uint32(x))
			candidate = e.table[currHash&tableMask]
			e.table[currHash&tableMask] = tableEntry{offset: e.cur + s, val: uint32(x)}

			offset := s - (candidate.offset - e.cur)
			if offset > maxMatchOffset || uint32(x) != candidate.val {
				cv = uint32(x >> 8)
				nextHash = hash(cv)
				s++
				break
			}
		}
	}

emitRemainder:
	if int(nextEmit) < len(src) {
		dst = emitLiteral(dst, src[nextEmit:])
	}
	e.cur += int32(len(src))
	e.prev = e.prev[:len(src)]
	copy(e.prev, src)
	return dst
}

func emitLiteral(dst []token, lit []byte) []token {
	for _, v := range lit {
		dst = append(dst, literalToken(uint32(v)))
	}
	return dst
}

// matchLen returns the match length between src[s:] and src[t:].
// t can be negative to indicate the match is starting in e.prev.
// We assume that src[s-4:s] and src[t-4:t] already match.
func (e *deflateFast) matchLen(s, t int32, src []byte) int32 {
	s1 := int(s) + maxMatchLength - 4
	if s1 > len(src) {
		s1 = len(src)
	}

	// If we are inside the current block
	if t >= 0 {
		b := src[t:]
		a := src[s:s1]
		b = b[:len(a)]
		// Extend the match to be as long as possible.
		for i := range a {
			if a[i] != b[i] {
				return int32(i)
			}
		}
		return int32(len(a))
	}

	// We found a match in the previous block.
	tp := int32(len(e.prev)) + t
	if tp < 0 {
		return 0
	}

	// Extend the match to be as long as possible.
	a := src[s:s1]
	b := e.prev[tp:]
	if len(b) > len(a) {
		b = b[:len(a)]
	}
	a = a[:len(b)]
	for i := range b {
		if a[i] != b[i] {
			return int32(i)
		}
	}

	// If we reached our limit, we matched everything we are
	// allowed to in the previous block and we return.
	n := int32(len(b))
	if int(s+n) == s1 {
		return n
	}

	// Continue looking for more matches in the current block.
	a = src[s+n : s1]
	b = src[:len(a)]
	for i := range a {
		if a[i] != b[i] {
			return int32(i) + n
		}
	}
	return int32(len(a)) + n
}

// Reset resets the encoding history.
// This ensures that no matches are made to the previous block.
func (e *deflateFast) reset() {
	e.prev = e.prev[:0]
	// Bump the offset, so all matches will fail distance check.
	e.cur += maxMatchOffset

	// Protect against e.cur wraparound.
	if e.cur > 1<<30 {
		e.resetAll()
	}
}

// resetAll resets the deflateFast struct and is only called in rare
// situations to prevent integer overflow. It manually resets each field
// to avoid causing large stack growth.
//
// See https://golang.org/issue/18636.
func (e *deflateFast) resetAll() {
	// This is equivalent to:
	//	*e = deflateFast{cur: maxStoreBlockSize, prev: e.prev[:0]}
	e.cur = maxStoreBlockSize
	e.prev = e.prev[:0]
	for i := range e.table {
		e.table[i] = tableEntry{}
	}
}
