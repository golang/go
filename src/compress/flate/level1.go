// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flate

// Level 1 uses a single small table with 5 byte hashes.
type fastEncL1 struct {
	fastGen
	table [tableSize]tableEntry
}

func (e *fastEncL1) Encode(dst *tokens, src []byte) {
	const (
		inputMargin            = 12 - 1
		minNonLiteralBlockSize = 1 + 1 + inputMargin
		hashBytes              = 5
	)

	// Protect against e.cur wraparound.
	for e.cur >= bufferReset {
		if len(e.hist) == 0 {
			for i := range e.table[:] {
				e.table[i] = tableEntry{}
			}
			e.cur = maxMatchOffset
			break
		}
		// Shift down everything in the table that isn't already too far away.
		minOff := e.cur + int32(len(e.hist)) - maxMatchOffset
		for i := range e.table[:] {
			v := e.table[i].offset
			if v <= minOff {
				v = 0
			} else {
				v = v - e.cur + maxMatchOffset
			}
			e.table[i].offset = v
		}
		e.cur = maxMatchOffset
	}

	s := e.addBlock(src)

	if len(src) < minNonLiteralBlockSize {
		// We do not fill the token table.
		// This will be picked up by caller.
		dst.n = uint16(len(src))
		return
	}

	// Override src
	src = e.hist

	// nextEmit is where in src the next emitLiterals should start from.
	nextEmit := s

	// sLimit is when to stop looking for offset/length copies. The inputMargin
	// lets us use a fast path for emitLiterals in the main loop, while we are
	// looking for copies.
	sLimit := int32(len(src) - inputMargin)

	cv := loadLE64(src, s)

	for {
		const skipLog = 5
		const doEvery = 2

		nextS := s
		var candidate tableEntry
		var t int32
		for {
			nextHash := hashLen(cv, tableBits, hashBytes)
			candidate = e.table[nextHash]
			nextS = s + doEvery + (s-nextEmit)>>skipLog
			if nextS > sLimit {
				goto emitRemainder
			}

			now := loadLE64(src, nextS)
			e.table[nextHash] = tableEntry{offset: s + e.cur}
			nextHash = hashLen(now, tableBits, hashBytes)
			t = candidate.offset - e.cur
			if s-t < maxMatchOffset && uint32(cv) == loadLE32(src, t) {
				e.table[nextHash] = tableEntry{offset: nextS + e.cur}
				break
			}

			// Do one right away...
			cv = now
			s = nextS
			nextS++
			candidate = e.table[nextHash]
			now >>= 8
			e.table[nextHash] = tableEntry{offset: s + e.cur}

			t = candidate.offset - e.cur
			if s-t < maxMatchOffset && uint32(cv) == loadLE32(src, t) {
				e.table[nextHash] = tableEntry{offset: nextS + e.cur}
				break
			}
			cv = now
			s = nextS
		}

		// A 4-byte match has been found. We'll later see if more than 4 bytes
		// match. But, prior to the match, src[nextEmit:s] are unmatched. Emit
		// them as literal bytes.
		for {
			// Invariant: we have a 4-byte match at s, and no need to emit any
			// literal bytes prior to s.

			// Extend the 4-byte match as long as possible.
			l := e.matchlenLong(int(s+4), int(t+4), src) + 4

			// Extend backwards
			for t > 0 && s > nextEmit && loadLE8(src, t-1) == loadLE8(src, s-1) {
				s--
				t--
				l++
			}
			if nextEmit < s {
				for _, v := range src[nextEmit:s] {
					dst.tokens[dst.n] = token(v)
					dst.litHist[v]++
					dst.n++
				}
			}

			// Save the match found. Same as 'dst.AddMatchLong(l, uint32(s-t-baseMatchOffset))'
			xOffset := uint32(s - t - baseMatchOffset)
			xLength := l
			oc := offsetCode(xOffset)
			xOffset |= oc << 16
			for xLength > 0 {
				xl := xLength
				if xl > 258 {
					if xl > 258+baseMatchLength {
						xl = 258
					} else {
						xl = 258 - baseMatchLength
					}
				}
				xLength -= xl
				xl -= baseMatchLength
				dst.extraHist[lengthCodes1[uint8(xl)]]++
				dst.offHist[oc]++
				dst.tokens[dst.n] = token(matchType | uint32(xl)<<lengthShift | xOffset)
				dst.n++
			}
			s += l
			nextEmit = s
			if nextS >= s {
				s = nextS + 1
			}
			if s >= sLimit {
				// Index first pair after match end.
				if int(s+l+8) < len(src) {
					cv := loadLE64(src, s)
					e.table[hashLen(cv, tableBits, hashBytes)] = tableEntry{offset: s + e.cur}
				}
				goto emitRemainder
			}

			// We could immediately start working at s now, but to improve
			// compression we first update the hash table at s-2 and at s. If
			// another emitCopy is not our next move, also calculate nextHash
			// at s+1. At least on GOARCH=amd64, these three hash calculations
			// are faster as one load64 call (with some shifts) instead of
			// three load32 calls.
			x := loadLE64(src, s-2)
			o := e.cur + s - 2
			prevHash := hashLen(x, tableBits, hashBytes)
			e.table[prevHash] = tableEntry{offset: o}
			x >>= 16
			currHash := hashLen(x, tableBits, hashBytes)
			candidate = e.table[currHash]
			e.table[currHash] = tableEntry{offset: o + 2}

			t = candidate.offset - e.cur
			if s-t > maxMatchOffset || uint32(x) != loadLE32(src, t) {
				cv = x >> 8
				s++
				break
			}
		}
	}

emitRemainder:
	if int(nextEmit) < len(src) {
		// If nothing was added, don't encode literals.
		if dst.n == 0 {
			return
		}
		emitLiterals(dst, src[nextEmit:])
	}
}
