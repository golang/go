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

func load32(b []byte, i int) uint32 {
	b = b[i : i+4 : len(b)] // Help the compiler eliminate bounds checks on the next line.
	return uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24
}

func load64(b []byte, i int) uint64 {
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

func encodeBestSpeed(dst []token, src []byte) []token {
	// This check isn't in the Snappy implementation, but there, the caller
	// instead of the callee handles this case.
	if len(src) < minNonLiteralBlockSize {
		return emitLiteral(dst, src)
	}

	// Initialize the hash table.
	//
	// The table element type is uint16, as s < sLimit and sLimit < len(src)
	// and len(src) <= maxStoreBlockSize and maxStoreBlockSize == 65535.
	var table [tableSize]uint16

	// sLimit is when to stop looking for offset/length copies. The inputMargin
	// lets us use a fast path for emitLiteral in the main loop, while we are
	// looking for copies.
	sLimit := len(src) - inputMargin

	// nextEmit is where in src the next emitLiteral should start from.
	nextEmit := 0

	// The encoded form must start with a literal, as there are no previous
	// bytes to copy, so we start looking for hash matches at s == 1.
	s := 1
	nextHash := hash(load32(src, s))

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
		skip := 32

		nextS := s
		candidate := 0
		for {
			s = nextS
			bytesBetweenHashLookups := skip >> 5
			nextS = s + bytesBetweenHashLookups
			skip += bytesBetweenHashLookups
			if nextS > sLimit {
				goto emitRemainder
			}
			candidate = int(table[nextHash&tableMask])
			table[nextHash&tableMask] = uint16(s)
			nextHash = hash(load32(src, nextS))
			// TODO: < should be <=, and add a test for that.
			if s-candidate < maxMatchOffset && load32(src, s) == load32(src, candidate) {
				break
			}
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
			base := s

			// Extend the 4-byte match as long as possible.
			//
			// This is an inlined version of Snappy's:
			//	s = extendMatch(src, candidate+4, s+4)
			s += 4
			s1 := base + maxMatchLength
			if s1 > len(src) {
				s1 = len(src)
			}
			for i := candidate + 4; s < s1 && src[i] == src[s]; i, s = i+1, s+1 {
			}

			// matchToken is flate's equivalent of Snappy's emitCopy.
			dst = append(dst, matchToken(uint32(s-base-baseMatchLength), uint32(base-candidate-baseMatchOffset)))
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
			prevHash := hash(uint32(x >> 0))
			table[prevHash&tableMask] = uint16(s - 1)
			currHash := hash(uint32(x >> 8))
			candidate = int(table[currHash&tableMask])
			table[currHash&tableMask] = uint16(s)
			// TODO: >= should be >, and add a test for that.
			if s-candidate >= maxMatchOffset || uint32(x>>8) != load32(src, candidate) {
				nextHash = hash(uint32(x >> 16))
				s++
				break
			}
		}
	}

emitRemainder:
	if nextEmit < len(src) {
		dst = emitLiteral(dst, src[nextEmit:])
	}
	return dst
}

func emitLiteral(dst []token, lit []byte) []token {
	for _, v := range lit {
		dst = append(dst, token(v))
	}
	return dst
}
