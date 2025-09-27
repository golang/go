// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flate

// Level 2 uses a similar algorithm to level 1, but with a larger table.
type fastEncL2 struct {
	fastGen
	table [bTableSize]tableEntry
}

func (e *fastEncL2) Encode(dst *tokens, src []byte) {
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
		// When should we start skipping if we haven't found matches in a long while.
		const skipLog = 5
		const doEvery = 2

		nextS := s
		var candidate tableEntry
		for {
			nextHash := hashLen(cv, bTableBits, hashBytes)
			s = nextS
			nextS = s + doEvery + (s-nextEmit)>>skipLog
			if nextS > sLimit {
				goto emitRemainder
			}
			candidate = e.table[nextHash]
			now := loadLE64(src, nextS)
			e.table[nextHash] = tableEntry{offset: s + e.cur}
			nextHash = hashLen(now, bTableBits, hashBytes)

			offset := s - (candidate.offset - e.cur)
			if offset < maxMatchOffset && uint32(cv) == loadLE32(src, candidate.offset-e.cur) {
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

			offset = s - (candidate.offset - e.cur)
			if offset < maxMatchOffset && uint32(cv) == loadLE32(src, candidate.offset-e.cur) {
				break
			}
			cv = now
		}

		// A 4-byte match has been found. We'll later see if more than 4 bytes match.
		for {
			// Extend the 4-byte match as long as possible.
			t := candidate.offset - e.cur
			l := e.matchlenLong(int(s+4), int(t+4), src) + 4

			// Extend backwards
			for t > 0 && s > nextEmit && src[t-1] == src[s-1] {
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

			dst.AddMatchLong(l, uint32(s-t-baseMatchOffset))
			s += l
			nextEmit = s
			if nextS >= s {
				s = nextS + 1
			}

			if s >= sLimit {
				// Index first pair after match end.
				if int(s+l+8) < len(src) {
					cv := loadLE64(src, s)
					e.table[hashLen(cv, bTableBits, hashBytes)] = tableEntry{offset: s + e.cur}
				}
				goto emitRemainder
			}

			// Store every second hash in-between, but offset by 1.
			for i := s - l + 2; i < s-5; i += 7 {
				x := loadLE64(src, i)
				nextHash := hashLen(x, bTableBits, hashBytes)
				e.table[nextHash] = tableEntry{offset: e.cur + i}
				// Skip one
				x >>= 16
				nextHash = hashLen(x, bTableBits, hashBytes)
				e.table[nextHash] = tableEntry{offset: e.cur + i + 2}
				// Skip one
				x >>= 16
				nextHash = hashLen(x, bTableBits, hashBytes)
				e.table[nextHash] = tableEntry{offset: e.cur + i + 4}
			}

			// We could immediately start working at s now, but to improve
			// compression we first update the hash table at s-2 to s. If
			// another emitCopy is not our next move, also calculate nextHash
			// at s+1.
			x := loadLE64(src, s-2)
			o := e.cur + s - 2
			prevHash := hashLen(x, bTableBits, hashBytes)
			prevHash2 := hashLen(x>>8, bTableBits, hashBytes)
			e.table[prevHash] = tableEntry{offset: o}
			e.table[prevHash2] = tableEntry{offset: o + 1}
			currHash := hashLen(x>>16, bTableBits, hashBytes)
			candidate = e.table[currHash]
			e.table[currHash] = tableEntry{offset: o + 2}

			offset := s - (candidate.offset - e.cur)
			if offset > maxMatchOffset || uint32(x>>16) != loadLE32(src, candidate.offset-e.cur) {
				cv = x >> 24
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
