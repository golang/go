// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flate

// Level 5 is similar to level 4, but for long matches two candidates are tested.
// Once a match is found, when it stops it will attempt to find a match that extends further.
type fastEncL5 struct {
	fastGen
	table  [tableSize]tableEntry
	bTable [tableSize]tableEntryPrev
}

func (e *fastEncL5) encode(dst *tokens, src []byte) {
	const (
		inputMargin            = 12 - 1
		minNonLiteralBlockSize = 1 + 1 + inputMargin
		hashShortBytes         = 4
	)

	// Protect against e.cur wraparound.
	for e.cur >= bufferReset {
		if len(e.hist) == 0 {
			clear(e.table[:])
			clear(e.bTable[:])
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
		for i := range e.bTable[:] {
			v := e.bTable[i]
			if v.cur.offset <= minOff {
				v.cur.offset = 0
				v.prev.offset = 0
			} else {
				v.cur.offset = v.cur.offset - e.cur + maxMatchOffset
				if v.prev.offset <= minOff {
					v.prev.offset = 0
				} else {
					v.prev.offset = v.prev.offset - e.cur + maxMatchOffset
				}
			}
			e.bTable[i] = v
		}
		e.cur = maxMatchOffset
	}

	s := e.addBlock(src)

	// This check isn't in the Snappy implementation, but there, the caller
	// instead of the callee handles this case.
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
		const skipLog = 6
		const doEvery = 1

		nextS := s
		var l int32
		var t int32
		for {
			nextHashS := hashLen(cv, tableBits, hashShortBytes)
			nextHashL := hashLen(cv, tableBits, hashLongBytes)

			s = nextS
			nextS = s + doEvery + (s-nextEmit)>>skipLog
			if nextS > sLimit {
				goto emitRemainder
			}
			// Fetch a short+long candidate
			sCandidate := e.table[nextHashS]
			lCandidate := e.bTable[nextHashL]
			next := loadLE64(src, nextS)
			entry := tableEntry{offset: s + e.cur}
			e.table[nextHashS] = entry
			eLong := &e.bTable[nextHashL]
			eLong.cur, eLong.prev = entry, eLong.cur

			nextHashS = hashLen(next, tableBits, hashShortBytes)
			nextHashL = hashLen(next, tableBits, hashLongBytes)

			t = lCandidate.cur.offset - e.cur
			if s-t < maxMatchOffset {
				if uint32(cv) == loadLE32(src, t) {
					// Store the next match
					e.table[nextHashS] = tableEntry{offset: nextS + e.cur}
					eLong := &e.bTable[nextHashL]
					eLong.cur, eLong.prev = tableEntry{offset: nextS + e.cur}, eLong.cur

					t2 := lCandidate.prev.offset - e.cur
					if s-t2 < maxMatchOffset && uint32(cv) == loadLE32(src, t2) {
						l = e.matchLenLimited(int(s+4), int(t+4), src) + 4
						ml1 := e.matchLenLimited(int(s+4), int(t2+4), src) + 4
						if ml1 > l {
							t = t2
							l = ml1
							break
						}
					}
					break
				}
				t = lCandidate.prev.offset - e.cur
				if s-t < maxMatchOffset && uint32(cv) == loadLE32(src, t) {
					// Store the next match
					e.table[nextHashS] = tableEntry{offset: nextS + e.cur}
					eLong := &e.bTable[nextHashL]
					eLong.cur, eLong.prev = tableEntry{offset: nextS + e.cur}, eLong.cur
					break
				}
			}

			t = sCandidate.offset - e.cur
			if s-t < maxMatchOffset && uint32(cv) == loadLE32(src, t) {
				// Found a 4 match...
				l = e.matchLenLimited(int(s+4), int(t+4), src) + 4
				lCandidate = e.bTable[nextHashL]
				// Store the next match

				e.table[nextHashS] = tableEntry{offset: nextS + e.cur}
				eLong := &e.bTable[nextHashL]
				eLong.cur, eLong.prev = tableEntry{offset: nextS + e.cur}, eLong.cur

				// If the next long is a candidate, use that...
				t2 := lCandidate.cur.offset - e.cur
				if nextS-t2 < maxMatchOffset {
					if loadLE32(src, t2) == uint32(next) {
						ml := e.matchLenLimited(int(nextS+4), int(t2+4), src) + 4
						if ml > l {
							t = t2
							s = nextS
							l = ml
							break
						}
					}
					// If the previous long is a candidate, use that...
					t2 = lCandidate.prev.offset - e.cur
					if nextS-t2 < maxMatchOffset && loadLE32(src, t2) == uint32(next) {
						ml := e.matchLenLimited(int(nextS+4), int(t2+4), src) + 4
						if ml > l {
							t = t2
							s = nextS
							l = ml
							break
						}
					}
				}
				break
			}
			cv = next
		}

		if l == 0 {
			// Extend the 4-byte match as long as possible.
			l = e.matchlenLong(int(s+4), int(t+4), src) + 4
		} else if l == maxMatchLength {
			l += e.matchlenLong(int(s+l), int(t+l), src)
		}

		// Try to locate a better match by checking the end of best match...
		if sAt := s + l; l < 30 && sAt < sLimit {
			// Allow some bytes at the beginning to mismatch.
			// Sweet spot is 2/3 bytes depending on input.
			// 3 is only a little better when it is but sometimes a lot worse.
			// The skipped bytes are tested in Extend backwards,
			// and still picked up as part of the match if they do.
			const skipBeginning = 2
			eLong := e.bTable[hashLen(loadLE64(src, sAt), tableBits, hashLongBytes)].cur.offset
			t2 := eLong - e.cur - l + skipBeginning
			s2 := s + skipBeginning
			off := s2 - t2
			if t2 >= 0 && off < maxMatchOffset && off > 0 {
				if l2 := e.matchlenLong(int(s2), int(t2), src); l2 > l {
					t = t2
					l = l2
					s = s2
				}
			}
		}

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
			goto emitRemainder
		}

		// Store every 3rd hash in-between.
		const hashEvery = 3
		i := s - l + 1
		if i < s-1 {
			cv := loadLE64(src, i)
			t := tableEntry{offset: i + e.cur}
			e.table[hashLen(cv, tableBits, hashShortBytes)] = t
			eLong := &e.bTable[hashLen(cv, tableBits, hashLongBytes)]
			eLong.cur, eLong.prev = t, eLong.cur

			// Do an long at i+1
			cv >>= 8
			t = tableEntry{offset: t.offset + 1}
			eLong = &e.bTable[hashLen(cv, tableBits, hashLongBytes)]
			eLong.cur, eLong.prev = t, eLong.cur

			// We only have enough bits for a short entry at i+2
			cv >>= 8
			t = tableEntry{offset: t.offset + 1}
			e.table[hashLen(cv, tableBits, hashShortBytes)] = t

			// Skip one - otherwise we risk hitting 's'
			i += 4
			for ; i < s-1; i += hashEvery {
				cv := loadLE64(src, i)
				t := tableEntry{offset: i + e.cur}
				t2 := tableEntry{offset: t.offset + 1}
				eLong := &e.bTable[hashLen(cv, tableBits, hashLongBytes)]
				eLong.cur, eLong.prev = t, eLong.cur
				e.table[hashLen(cv>>8, tableBits, hashShortBytes)] = t2
			}
		}

		// We could immediately start working at s now, but to improve
		// compression we first update the hash table at s-1 and at s.
		x := loadLE64(src, s-1)
		o := e.cur + s - 1
		prevHashS := hashLen(x, tableBits, hashShortBytes)
		prevHashL := hashLen(x, tableBits, hashLongBytes)
		e.table[prevHashS] = tableEntry{offset: o}
		eLong := &e.bTable[prevHashL]
		eLong.cur, eLong.prev = tableEntry{offset: o}, eLong.cur
		cv = x >> 8
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
