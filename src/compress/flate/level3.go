// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flate

// Level 3 uses a similar algorithm to level 2, with a smaller table,
// but will check up two candidates for each iteration with more
// entries added to the table.
type fastEncL3 struct {
	fastGen
	table [1 << 16]tableEntryPrev
}

func (e *fastEncL3) encode(dst *tokens, src []byte) {
	const (
		inputMargin            = 12 - 1
		minNonLiteralBlockSize = 1 + 1 + inputMargin
		tableBits              = 16
		hashBytes              = 5
	)

	// Protect against e.cur wraparound.
	for e.cur >= bufferReset {
		if len(e.hist) == 0 {
			clear(e.table[:])
			e.cur = maxMatchOffset
			break
		}
		// Shift down everything in the table that isn't already too far away.
		minOff := e.cur + int32(len(e.hist)) - maxMatchOffset
		for i := range e.table[:] {
			v := e.table[i]
			if v.cur.offset <= minOff {
				v.cur.offset = 0
			} else {
				v.cur.offset = v.cur.offset - e.cur + maxMatchOffset
			}
			if v.prev.offset <= minOff {
				v.prev.offset = 0
			} else {
				v.prev.offset = v.prev.offset - e.cur + maxMatchOffset
			}
			e.table[i] = v
		}
		e.cur = maxMatchOffset
	}

	s := e.addBlock(src)

	// Skip if too small.
	if len(src) < minNonLiteralBlockSize {
		// We do not fill the token table.
		// This will be picked up by caller.
		dst.n = uint16(len(src))
		return
	}

	// Override src
	src = e.hist
	nextEmit := s

	// sLimit is when to stop looking for offset/length copies. The inputMargin
	// lets us use a fast path for emitLiterals in the main loop, while we are
	// looking for copies.
	sLimit := int32(len(src) - inputMargin)

	// nextEmit is where in src the next emitLiterals should start from.
	cv := loadLE64(src, s)
	for {
		const skipLog = 7
		nextS := s
		var candidate tableEntry
		for {
			nextHash := hashLen(cv, tableBits, hashBytes)
			s = nextS
			nextS = s + 1 + (s-nextEmit)>>skipLog
			if nextS > sLimit {
				goto emitRemainder
			}
			candidates := e.table[nextHash]
			now := loadLE64(src, nextS)

			// Safe offset distance until s + 4...
			minOffset := e.cur + s - (maxMatchOffset - 4)
			e.table[nextHash] = tableEntryPrev{prev: candidates.cur, cur: tableEntry{offset: s + e.cur}}

			// Check both candidates
			candidate = candidates.cur
			if candidate.offset < minOffset {
				cv = now
				// Previous will also be invalid, we have nothing.
				continue
			}

			if uint32(cv) == loadLE32(src, candidate.offset-e.cur) {
				if candidates.prev.offset < minOffset || uint32(cv) != loadLE32(src, candidates.prev.offset-e.cur) {
					break
				}
				// Both match and are valid, pick longest.
				offset := s - (candidate.offset - e.cur)
				o2 := s - (candidates.prev.offset - e.cur)
				l1, l2 := matchLen(src[s+4:], src[s-offset+4:]), matchLen(src[s+4:], src[s-o2+4:])
				if l2 > l1 {
					candidate = candidates.prev
				}
				break
			} else {
				// We only check if value mismatches.
				// Offset will always be invalid in other cases.
				candidate = candidates.prev
				if candidate.offset > minOffset && uint32(cv) == loadLE32(src, candidate.offset-e.cur) {
					break
				}
			}
			cv = now
		}

		for {
			// Extend the 4-byte match as long as possible.
			//
			t := candidate.offset - e.cur
			l := e.matchlenLong(int(s+4), int(t+4), src) + 4

			// Extend backwards
			for t > 0 && s > nextEmit && src[t-1] == src[s-1] {
				s--
				t--
				l++
			}
			// Emit literals.
			if nextEmit < s {
				for _, v := range src[nextEmit:s] {
					dst.tokens[dst.n] = token(v)
					dst.litHist[v]++
					dst.n++
				}
			}

			// Emit match.
			dst.AddMatchLong(l, uint32(s-t-baseMatchOffset))
			s += l
			nextEmit = s
			if nextS >= s {
				s = nextS + 1
			}

			if s >= sLimit {
				t += l
				// Index first pair after match end.
				if int(t+8) < len(src) && t > 0 {
					cv = loadLE64(src, t)
					nextHash := hashLen(cv, tableBits, hashBytes)
					e.table[nextHash] = tableEntryPrev{
						prev: e.table[nextHash].cur,
						cur:  tableEntry{offset: e.cur + t},
					}
				}
				goto emitRemainder
			}

			// Store every 5th hash in-between.
			for i := s - l + 2; i < s-5; i += 6 {
				nextHash := hashLen(loadLE64(src, i), tableBits, hashBytes)
				e.table[nextHash] = tableEntryPrev{
					prev: e.table[nextHash].cur,
					cur:  tableEntry{offset: e.cur + i}}
			}
			// We could immediately start working at s now, but to improve
			// compression we first update the hash table at s-2 to s.
			x := loadLE64(src, s-2)
			prevHash := hashLen(x, tableBits, hashBytes)

			e.table[prevHash] = tableEntryPrev{
				prev: e.table[prevHash].cur,
				cur:  tableEntry{offset: e.cur + s - 2},
			}
			x >>= 8
			prevHash = hashLen(x, tableBits, hashBytes)

			e.table[prevHash] = tableEntryPrev{
				prev: e.table[prevHash].cur,
				cur:  tableEntry{offset: e.cur + s - 1},
			}
			x >>= 8
			currHash := hashLen(x, tableBits, hashBytes)
			candidates := e.table[currHash]
			cv = x
			e.table[currHash] = tableEntryPrev{
				prev: candidates.cur,
				cur:  tableEntry{offset: s + e.cur},
			}

			// Check both candidates
			candidate = candidates.cur
			minOffset := e.cur + s - (maxMatchOffset - 4)

			if candidate.offset > minOffset {
				if uint32(cv) == loadLE32(src, candidate.offset-e.cur) {
					// Found a match...
					continue
				}
				candidate = candidates.prev
				if candidate.offset > minOffset && uint32(cv) == loadLE32(src, candidate.offset-e.cur) {
					// Match at prev...
					continue
				}
			}
			cv = x >> 8
			s++
			break
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
