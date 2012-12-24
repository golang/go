// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package norm

import "unicode/utf8"

const (
	maxCombiningChars = 30
	maxBufferSize     = maxCombiningChars + 2 // +1 to hold starter +1 to hold CGJ
	maxBackRunes      = maxCombiningChars - 1
	maxNFCExpansion   = 3  // NFC(0x1D160)
	maxNFKCExpansion  = 18 // NFKC(0xFDFA)

	maxByteBufferSize = utf8.UTFMax * maxBufferSize // 128
)

// reorderBuffer is used to normalize a single segment.  Characters inserted with
// insert are decomposed and reordered based on CCC. The compose method can
// be used to recombine characters.  Note that the byte buffer does not hold
// the UTF-8 characters in order.  Only the rune array is maintained in sorted
// order. flush writes the resulting segment to a byte array.
type reorderBuffer struct {
	rune  [maxBufferSize]Properties // Per character info.
	byte  [maxByteBufferSize]byte   // UTF-8 buffer. Referenced by runeInfo.pos.
	nrune int                       // Number of runeInfos.
	nbyte uint8                     // Number or bytes.
	f     formInfo

	src      input
	nsrc     int
	tmpBytes input
}

func (rb *reorderBuffer) init(f Form, src []byte) {
	rb.f = *formTable[f]
	rb.src.setBytes(src)
	rb.nsrc = len(src)
}

func (rb *reorderBuffer) initString(f Form, src string) {
	rb.f = *formTable[f]
	rb.src.setString(src)
	rb.nsrc = len(src)
}

// reset discards all characters from the buffer.
func (rb *reorderBuffer) reset() {
	rb.nrune = 0
	rb.nbyte = 0
}

// flush appends the normalized segment to out and resets rb.
func (rb *reorderBuffer) flush(out []byte) []byte {
	for i := 0; i < rb.nrune; i++ {
		start := rb.rune[i].pos
		end := start + rb.rune[i].size
		out = append(out, rb.byte[start:end]...)
	}
	rb.reset()
	return out
}

// flushCopy copies the normalized segment to buf and resets rb.
// It returns the number of bytes written to buf.
func (rb *reorderBuffer) flushCopy(buf []byte) int {
	p := 0
	for i := 0; i < rb.nrune; i++ {
		runep := rb.rune[i]
		p += copy(buf[p:], rb.byte[runep.pos:runep.pos+runep.size])
	}
	rb.reset()
	return p
}

// insertOrdered inserts a rune in the buffer, ordered by Canonical Combining Class.
// It returns false if the buffer is not large enough to hold the rune.
// It is used internally by insert and insertString only.
func (rb *reorderBuffer) insertOrdered(info Properties) bool {
	n := rb.nrune
	if n >= maxCombiningChars+1 {
		return false
	}
	b := rb.rune[:]
	cc := info.ccc
	if cc > 0 {
		// Find insertion position + move elements to make room.
		for ; n > 0; n-- {
			if b[n-1].ccc <= cc {
				break
			}
			b[n] = b[n-1]
		}
	}
	rb.nrune += 1
	pos := uint8(rb.nbyte)
	rb.nbyte += utf8.UTFMax
	info.pos = pos
	b[n] = info
	return true
}

// insert inserts the given rune in the buffer ordered by CCC.
// It returns true if the buffer was large enough to hold the decomposed rune.
func (rb *reorderBuffer) insert(src input, i int, info Properties) bool {
	if rune := src.hangul(i); rune != 0 {
		return rb.decomposeHangul(rune)
	}
	if info.hasDecomposition() {
		return rb.insertDecomposed(info.Decomposition())
	}
	return rb.insertSingle(src, i, info)
}

// insertDecomposed inserts an entry in to the reorderBuffer for each rune
// in dcomp.  dcomp must be a sequence of decomposed UTF-8-encoded runes.
func (rb *reorderBuffer) insertDecomposed(dcomp []byte) bool {
	saveNrune, saveNbyte := rb.nrune, rb.nbyte
	rb.tmpBytes.setBytes(dcomp)
	for i := 0; i < len(dcomp); {
		info := rb.f.info(rb.tmpBytes, i)
		pos := rb.nbyte
		if !rb.insertOrdered(info) {
			rb.nrune, rb.nbyte = saveNrune, saveNbyte
			return false
		}
		i += copy(rb.byte[pos:], dcomp[i:i+int(info.size)])
	}
	return true
}

// insertSingle inserts an entry in the reorderBuffer for the rune at
// position i. info is the runeInfo for the rune at position i.
func (rb *reorderBuffer) insertSingle(src input, i int, info Properties) bool {
	// insertOrder changes nbyte
	pos := rb.nbyte
	if !rb.insertOrdered(info) {
		return false
	}
	src.copySlice(rb.byte[pos:], i, i+int(info.size))
	return true
}

// appendRune inserts a rune at the end of the buffer. It is used for Hangul.
func (rb *reorderBuffer) appendRune(r rune) {
	bn := rb.nbyte
	sz := utf8.EncodeRune(rb.byte[bn:], rune(r))
	rb.nbyte += utf8.UTFMax
	rb.rune[rb.nrune] = Properties{pos: bn, size: uint8(sz)}
	rb.nrune++
}

// assignRune sets a rune at position pos. It is used for Hangul and recomposition.
func (rb *reorderBuffer) assignRune(pos int, r rune) {
	bn := rb.rune[pos].pos
	sz := utf8.EncodeRune(rb.byte[bn:], rune(r))
	rb.rune[pos] = Properties{pos: bn, size: uint8(sz)}
}

// runeAt returns the rune at position n. It is used for Hangul and recomposition.
func (rb *reorderBuffer) runeAt(n int) rune {
	inf := rb.rune[n]
	r, _ := utf8.DecodeRune(rb.byte[inf.pos : inf.pos+inf.size])
	return r
}

// bytesAt returns the UTF-8 encoding of the rune at position n.
// It is used for Hangul and recomposition.
func (rb *reorderBuffer) bytesAt(n int) []byte {
	inf := rb.rune[n]
	return rb.byte[inf.pos : int(inf.pos)+int(inf.size)]
}

// For Hangul we combine algorithmically, instead of using tables.
const (
	hangulBase  = 0xAC00 // UTF-8(hangulBase) -> EA B0 80
	hangulBase0 = 0xEA
	hangulBase1 = 0xB0
	hangulBase2 = 0x80

	hangulEnd  = hangulBase + jamoLVTCount // UTF-8(0xD7A4) -> ED 9E A4
	hangulEnd0 = 0xED
	hangulEnd1 = 0x9E
	hangulEnd2 = 0xA4

	jamoLBase  = 0x1100 // UTF-8(jamoLBase) -> E1 84 00
	jamoLBase0 = 0xE1
	jamoLBase1 = 0x84
	jamoLEnd   = 0x1113
	jamoVBase  = 0x1161
	jamoVEnd   = 0x1176
	jamoTBase  = 0x11A7
	jamoTEnd   = 0x11C3

	jamoTCount   = 28
	jamoVCount   = 21
	jamoVTCount  = 21 * 28
	jamoLVTCount = 19 * 21 * 28
)

const hangulUTF8Size = 3

func isHangul(b []byte) bool {
	if len(b) < hangulUTF8Size {
		return false
	}
	b0 := b[0]
	if b0 < hangulBase0 {
		return false
	}
	b1 := b[1]
	switch {
	case b0 == hangulBase0:
		return b1 >= hangulBase1
	case b0 < hangulEnd0:
		return true
	case b0 > hangulEnd0:
		return false
	case b1 < hangulEnd1:
		return true
	}
	return b1 == hangulEnd1 && b[2] < hangulEnd2
}

func isHangulString(b string) bool {
	if len(b) < hangulUTF8Size {
		return false
	}
	b0 := b[0]
	if b0 < hangulBase0 {
		return false
	}
	b1 := b[1]
	switch {
	case b0 == hangulBase0:
		return b1 >= hangulBase1
	case b0 < hangulEnd0:
		return true
	case b0 > hangulEnd0:
		return false
	case b1 < hangulEnd1:
		return true
	}
	return b1 == hangulEnd1 && b[2] < hangulEnd2
}

// Caller must ensure len(b) >= 2.
func isJamoVT(b []byte) bool {
	// True if (rune & 0xff00) == jamoLBase
	return b[0] == jamoLBase0 && (b[1]&0xFC) == jamoLBase1
}

func isHangulWithoutJamoT(b []byte) bool {
	c, _ := utf8.DecodeRune(b)
	c -= hangulBase
	return c < jamoLVTCount && c%jamoTCount == 0
}

// decomposeHangul writes the decomposed Hangul to buf and returns the number
// of bytes written.  len(buf) should be at least 9.
func decomposeHangul(buf []byte, r rune) int {
	const JamoUTF8Len = 3
	r -= hangulBase
	x := r % jamoTCount
	r /= jamoTCount
	utf8.EncodeRune(buf, jamoLBase+r/jamoVCount)
	utf8.EncodeRune(buf[JamoUTF8Len:], jamoVBase+r%jamoVCount)
	if x != 0 {
		utf8.EncodeRune(buf[2*JamoUTF8Len:], jamoTBase+x)
		return 3 * JamoUTF8Len
	}
	return 2 * JamoUTF8Len
}

// decomposeHangul algorithmically decomposes a Hangul rune into
// its Jamo components.
// See http://unicode.org/reports/tr15/#Hangul for details on decomposing Hangul.
func (rb *reorderBuffer) decomposeHangul(r rune) bool {
	b := rb.rune[:]
	n := rb.nrune
	if n+3 > len(b) {
		return false
	}
	r -= hangulBase
	x := r % jamoTCount
	r /= jamoTCount
	rb.appendRune(jamoLBase + r/jamoVCount)
	rb.appendRune(jamoVBase + r%jamoVCount)
	if x != 0 {
		rb.appendRune(jamoTBase + x)
	}
	return true
}

// combineHangul algorithmically combines Jamo character components into Hangul.
// See http://unicode.org/reports/tr15/#Hangul for details on combining Hangul.
func (rb *reorderBuffer) combineHangul(s, i, k int) {
	b := rb.rune[:]
	bn := rb.nrune
	for ; i < bn; i++ {
		cccB := b[k-1].ccc
		cccC := b[i].ccc
		if cccB == 0 {
			s = k - 1
		}
		if s != k-1 && cccB >= cccC {
			// b[i] is blocked by greater-equal cccX below it
			b[k] = b[i]
			k++
		} else {
			l := rb.runeAt(s) // also used to compare to hangulBase
			v := rb.runeAt(i) // also used to compare to jamoT
			switch {
			case jamoLBase <= l && l < jamoLEnd &&
				jamoVBase <= v && v < jamoVEnd:
				// 11xx plus 116x to LV
				rb.assignRune(s, hangulBase+
					(l-jamoLBase)*jamoVTCount+(v-jamoVBase)*jamoTCount)
			case hangulBase <= l && l < hangulEnd &&
				jamoTBase < v && v < jamoTEnd &&
				((l-hangulBase)%jamoTCount) == 0:
				// ACxx plus 11Ax to LVT
				rb.assignRune(s, l+v-jamoTBase)
			default:
				b[k] = b[i]
				k++
			}
		}
	}
	rb.nrune = k
}

// compose recombines the runes in the buffer.
// It should only be used to recompose a single segment, as it will not
// handle alternations between Hangul and non-Hangul characters correctly.
func (rb *reorderBuffer) compose() {
	// UAX #15, section X5 , including Corrigendum #5
	// "In any character sequence beginning with starter S, a character C is
	//  blocked from S if and only if there is some character B between S
	//  and C, and either B is a starter or it has the same or higher
	//  combining class as C."
	bn := rb.nrune
	if bn == 0 {
		return
	}
	k := 1
	b := rb.rune[:]
	for s, i := 0, 1; i < bn; i++ {
		if isJamoVT(rb.bytesAt(i)) {
			// Redo from start in Hangul mode. Necessary to support
			// U+320E..U+321E in NFKC mode.
			rb.combineHangul(s, i, k)
			return
		}
		ii := b[i]
		// We can only use combineForward as a filter if we later
		// get the info for the combined character. This is more
		// expensive than using the filter. Using combinesBackward()
		// is safe.
		if ii.combinesBackward() {
			cccB := b[k-1].ccc
			cccC := ii.ccc
			blocked := false // b[i] blocked by starter or greater or equal CCC?
			if cccB == 0 {
				s = k - 1
			} else {
				blocked = s != k-1 && cccB >= cccC
			}
			if !blocked {
				combined := combine(rb.runeAt(s), rb.runeAt(i))
				if combined != 0 {
					rb.assignRune(s, combined)
					continue
				}
			}
		}
		b[k] = b[i]
		k++
	}
	rb.nrune = k
}
