// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flate

// Level 6 extends level 5, but does "repeat offset" check,
// as well as adding more hash entries to the tables.
type fastEncL6 struct {
	fastGen
	table  [tableSize]tableEntry
	bTable [tableSize]tableEntryPrev
}

func (e *fastEncL6) Encode(dst *tokens, src []byte) {
	const (
		inputMargin            = 12 - 1
		minNonLiteralBlockSize = 1 + 1 + inputMargin
		hashShortBytes         = 4
	)

	// Protect against e.cur wraparound.
	for e.cur >= bufferReset {
		if len(e.hist) == 0 {
			for i := range e.table[:] {
				e.table[i] = tableEntry{}
			}
			for i := range e.bTable[:] {
				e.bTable[i] = tableEntryPrev{}
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
		for i := range e.bTable[:] {
			v := e.bTable[i]
			if v.Cur.offset <= minOff {
				v.Cur.offset = 0
				v.Prev.offset = 0
			} else {
				v.Cur.offset = v.Cur.offset - e.cur + maxMatchOffset
				if v.Prev.offset <= minOff {
					v.Prev.offset = 0
				} else {
					v.Prev.offset = v.Prev.offset - e.cur + maxMatchOffset
				}
			}
			e.bTable[i] = v
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
	// Repeat MUST be > 1 and within range
	repeat := int32(1)
	for {
		const skipLog = 7
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
			eLong.Cur, eLong.Prev = entry, eLong.Cur

			// Calculate hashes of 'next'
			nextHashS = hashLen(next, tableBits, hashShortBytes)
			nextHashL = hashLen(next, tableBits, hashLongBytes)

			t = lCandidate.Cur.offset - e.cur
			if s-t < maxMatchOffset {
				if uint32(cv) == loadLE32(src, t) {
					// Long candidate matches at least 4 bytes.

					// Store the next match
					e.table[nextHashS] = tableEntry{offset: nextS + e.cur}
					eLong := &e.bTable[nextHashL]
					eLong.Cur, eLong.Prev = tableEntry{offset: nextS + e.cur}, eLong.Cur

					// Check the previous long candidate as well.
					t2 := lCandidate.Prev.offset - e.cur
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
				// Current value did not match, but check if previous long value does.
				t = lCandidate.Prev.offset - e.cur
				if s-t < maxMatchOffset && uint32(cv) == loadLE32(src, t) {
					// Store the next match
					e.table[nextHashS] = tableEntry{offset: nextS + e.cur}
					eLong := &e.bTable[nextHashL]
					eLong.Cur, eLong.Prev = tableEntry{offset: nextS + e.cur}, eLong.Cur
					break
				}
			}

			t = sCandidate.offset - e.cur
			if s-t < maxMatchOffset && uint32(cv) == loadLE32(src, t) {
				// Found a 4 match...
				l = e.matchLenLimited(int(s+4), int(t+4), src) + 4

				// Look up next long candidate (at nextS)
				lCandidate = e.bTable[nextHashL]

				// Store the next match
				e.table[nextHashS] = tableEntry{offset: nextS + e.cur}
				eLong := &e.bTable[nextHashL]
				eLong.Cur, eLong.Prev = tableEntry{offset: nextS + e.cur}, eLong.Cur

				// Check repeat at s + repOff
				const repOff = 1
				t2 := s - repeat + repOff
				if loadLE32(src, t2) == uint32(cv>>(8*repOff)) {
					ml := e.matchLenLimited(int(s+4+repOff), int(t2+4), src) + 4
					if ml > l {
						t = t2
						l = ml
						s += repOff
						// Not worth checking more.
						break
					}
				}

				// If the next long is a candidate, use that...
				t2 = lCandidate.Cur.offset - e.cur
				if nextS-t2 < maxMatchOffset {
					if loadLE32(src, t2) == uint32(next) {
						ml := e.matchLenLimited(int(nextS+4), int(t2+4), src) + 4
						if ml > l {
							t = t2
							s = nextS
							l = ml
							// This is ok, but check previous as well.
						}
					}
					// If the previous long is a candidate, use that...
					t2 = lCandidate.Prev.offset - e.cur
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

		// Extend the 4-byte match as long as possible.
		if l == 0 {
			l = e.matchlenLong(int(s+4), int(t+4), src) + 4
		} else if l == maxMatchLength {
			l += e.matchlenLong(int(s+l), int(t+l), src)
		}

		// Try to locate a better match by checking the end-of-match...
		if sAt := s + l; sAt < sLimit {
			// Allow some bytes at the beginning to mismatch.
			// Sweet spot is 2/3 bytes depending on input.
			// 3 is only a little better when it is but sometimes a lot worse.
			// The skipped bytes are tested in extend backwards,
			// and still picked up as part of the match if they do.
			const skipBeginning = 2
			eLong := &e.bTable[hashLen(loadLE64(src, sAt), tableBits, hashLongBytes)]
			// Test current
			t2 := eLong.Cur.offset - e.cur - l + skipBeginning
			s2 := s + skipBeginning
			off := s2 - t2
			if off < maxMatchOffset {
				if off > 0 && t2 >= 0 {
					if l2 := e.matchlenLong(int(s2), int(t2), src); l2 > l {
						t = t2
						l = l2
						s = s2
					}
				}
				// Test previous entry:
				t2 = eLong.Prev.offset - e.cur - l + skipBeginning
				off := s2 - t2
				if off > 0 && off < maxMatchOffset && t2 >= 0 {
					if l2 := e.matchlenLong(int(s2), int(t2), src); l2 > l {
						t = t2
						l = l2
						s = s2
					}
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
		repeat = s - t
		s += l
		nextEmit = s
		if nextS >= s {
			s = nextS + 1
		}

		if s >= sLimit {
			// Index after match end.
			for i := nextS + 1; i < int32(len(src))-8; i += 2 {
				cv := loadLE64(src, i)
				e.table[hashLen(cv, tableBits, hashShortBytes)] = tableEntry{offset: i + e.cur}
				eLong := &e.bTable[hashLen(cv, tableBits, hashLongBytes)]
				eLong.Cur, eLong.Prev = tableEntry{offset: i + e.cur}, eLong.Cur
			}
			goto emitRemainder
		}

		// Store every long hash in-between and every second short.
		for i := nextS + 1; i < s-1; i += 2 {
			cv := loadLE64(src, i)
			t := tableEntry{offset: i + e.cur}
			t2 := tableEntry{offset: t.offset + 1}
			eLong := &e.bTable[hashLen(cv, tableBits, hashLongBytes)]
			eLong2 := &e.bTable[hashLen(cv>>8, tableBits, hashLongBytes)]
			e.table[hashLen(cv, tableBits, hashShortBytes)] = t
			eLong.Cur, eLong.Prev = t, eLong.Cur
			eLong2.Cur, eLong2.Prev = t2, eLong2.Cur
		}
		cv = loadLE64(src, s)
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
