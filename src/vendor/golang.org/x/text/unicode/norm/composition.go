// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package norm

import "unicode/utf8"

const (
	maxNonStarters = 30
	// The maximum number of characters needed for a buffer is
	// maxNonStarters + 1 for the starter + 1 for the GCJ
	maxBufferSize    = maxNonStarters + 2
	maxNFCExpansion  = 3  // NFC(0x1D160)
	maxNFKCExpansion = 18 // NFKC(0xFDFA)

	maxByteBufferSize = utf8.UTFMax * maxBufferSize // 128
)

// ssState is used for reporting the segment state after inserting a rune.
// It is returned by streamSafe.next.
type ssState int

const (
	// Indicates a rune was successfully added to the segment.
	ssSuccess ssState = iota
	// Indicates a rune starts a new segment and should not be added.
	ssStarter
	// Indicates a rune caused a segment overflow and a CGJ should be inserted.
	ssOverflow
)

// streamSafe implements the policy of when a CGJ should be inserted.
type streamSafe uint8

// first inserts the first rune of a segment. It is a faster version of next if
// it is known p represents the first rune in a segment.
func (ss *streamSafe) first(p Properties) {
	*ss = streamSafe(p.nTrailingNonStarters())
}

// insert returns a ssState value to indicate whether a rune represented by p
// can be inserted.
func (ss *streamSafe) next(p Properties) ssState {
	if *ss > maxNonStarters {
		panic("streamSafe was not reset")
	}
	n := p.nLeadingNonStarters()
	if *ss += streamSafe(n); *ss > maxNonStarters {
		*ss = 0
		return ssOverflow
	}
	// The Stream-Safe Text Processing prescribes that the counting can stop
	// as soon as a starter is encountered. However, there are some starters,
	// like Jamo V and T, that can combine with other runes, leaving their
	// successive non-starters appended to the previous, possibly causing an
	// overflow. We will therefore consider any rune with a non-zero nLead to
	// be a non-starter. Note that it always hold that if nLead > 0 then
	// nLead == nTrail.
	if n == 0 {
		*ss = streamSafe(p.nTrailingNonStarters())
		return ssStarter
	}
	return ssSuccess
}

// backwards is used for checking for overflow and segment starts
// when traversing a string backwards. Users do not need to call first
// for the first rune. The state of the streamSafe retains the count of
// the non-starters loaded.
func (ss *streamSafe) backwards(p Properties) ssState {
	if *ss > maxNonStarters {
		panic("streamSafe was not reset")
	}
	c := *ss + streamSafe(p.nTrailingNonStarters())
	if c > maxNonStarters {
		return ssOverflow
	}
	*ss = c
	if p.nLeadingNonStarters() == 0 {
		return ssStarter
	}
	return ssSuccess
}

func (ss streamSafe) isMax() bool {
	return ss == maxNonStarters
}

// GraphemeJoiner is inserted after maxNonStarters non-starter runes.
const GraphemeJoiner = "\u034F"

// reorderBuffer is used to normalize a single segment.  Characters inserted with
// insert are decomposed and reordered based on CCC. The compose method can
// be used to recombine characters.  Note that the byte buffer does not hold
// the UTF-8 characters in order.  Only the rune array is maintained in sorted
// order. flush writes the resulting segment to a byte array.
type reorderBuffer struct {
	rune  [maxBufferSize]Properties // Per character info.
	byte  [maxByteBufferSize]byte   // UTF-8 buffer. Referenced by runeInfo.pos.
	nbyte uint8                     // Number or bytes.
	ss    streamSafe                // For limiting length of non-starter sequence.
	nrune int                       // Number of runeInfos.
	f     formInfo

	src      input
	nsrc     int
	tmpBytes input

	out    []byte
	flushF func(*reorderBuffer) bool
}

func (rb *reorderBuffer) init(f Form, src []byte) {
	rb.f = *formTable[f]
	rb.src.setBytes(src)
	rb.nsrc = len(src)
	rb.ss = 0
}

func (rb *reorderBuffer) initString(f Form, src string) {
	rb.f = *formTable[f]
	rb.src.setString(src)
	rb.nsrc = len(src)
	rb.ss = 0
}

func (rb *reorderBuffer) setFlusher(out []byte, f func(*reorderBuffer) bool) {
	rb.out = out
	rb.flushF = f
}

// reset discards all characters from the buffer.
func (rb *reorderBuffer) reset() {
	rb.nrune = 0
	rb.nbyte = 0
}

func (rb *reorderBuffer) doFlush() bool {
	if rb.f.composing {
		rb.compose()
	}
	res := rb.flushF(rb)
	rb.reset()
	return res
}

// appendFlush appends the normalized segment to rb.out.
func appendFlush(rb *reorderBuffer) bool {
	for i := 0; i < rb.nrune; i++ {
		start := rb.rune[i].pos
		end := start + rb.rune[i].size
		rb.out = append(rb.out, rb.byte[start:end]...)
	}
	return true
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
func (rb *reorderBuffer) insertOrdered(info Properties) {
	n := rb.nrune
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
}

// insertErr is an error code returned by insert. Using this type instead
// of error improves performance up to 20% for many of the benchmarks.
type insertErr int

const (
	iSuccess insertErr = -iota
	iShortDst
	iShortSrc
)

// insertFlush inserts the given rune in the buffer ordered by CCC.
// If a decomposition with multiple segments are encountered, they leading
// ones are flushed.
// It returns a non-zero error code if the rune was not inserted.
func (rb *reorderBuffer) insertFlush(src input, i int, info Properties) insertErr {
	if rune := src.hangul(i); rune != 0 {
		rb.decomposeHangul(rune)
		return iSuccess
	}
	if info.hasDecomposition() {
		return rb.insertDecomposed(info.Decomposition())
	}
	rb.insertSingle(src, i, info)
	return iSuccess
}

// insertUnsafe inserts the given rune in the buffer ordered by CCC.
// It is assumed there is sufficient space to hold the runes. It is the
// responsibility of the caller to ensure this. This can be done by checking
// the state returned by the streamSafe type.
func (rb *reorderBuffer) insertUnsafe(src input, i int, info Properties) {
	if rune := src.hangul(i); rune != 0 {
		rb.decomposeHangul(rune)
	}
	if info.hasDecomposition() {
		// TODO: inline.
		rb.insertDecomposed(info.Decomposition())
	} else {
		rb.insertSingle(src, i, info)
	}
}

// insertDecomposed inserts an entry in to the reorderBuffer for each rune
// in dcomp. dcomp must be a sequence of decomposed UTF-8-encoded runes.
// It flushes the buffer on each new segment start.
func (rb *reorderBuffer) insertDecomposed(dcomp []byte) insertErr {
	rb.tmpBytes.setBytes(dcomp)
	// As the streamSafe accounting already handles the counting for modifiers,
	// we don't have to call next. However, we do need to keep the accounting
	// intact when flushing the buffer.
	for i := 0; i < len(dcomp); {
		info := rb.f.info(rb.tmpBytes, i)
		if info.BoundaryBefore() && rb.nrune > 0 && !rb.doFlush() {
			return iShortDst
		}
		i += copy(rb.byte[rb.nbyte:], dcomp[i:i+int(info.size)])
		rb.insertOrdered(info)
	}
	return iSuccess
}

// insertSingle inserts an entry in the reorderBuffer for the rune at
// position i. info is the runeInfo for the rune at position i.
func (rb *reorderBuffer) insertSingle(src input, i int, info Properties) {
	src.copySlice(rb.byte[rb.nbyte:], i, i+int(info.size))
	rb.insertOrdered(info)
}

// insertCGJ inserts a Combining Grapheme Joiner (0x034f) into rb.
func (rb *reorderBuffer) insertCGJ() {
	rb.insertSingle(input{str: GraphemeJoiner}, 0, Properties{size: uint8(len(GraphemeJoiner))})
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
func (rb *reorderBuffer) decomposeHangul(r rune) {
	r -= hangulBase
	x := r % jamoTCount
	r /= jamoTCount
	rb.appendRune(jamoLBase + r/jamoVCount)
	rb.appendRune(jamoVBase + r%jamoVCount)
	if x != 0 {
		rb.appendRune(jamoTBase + x)
	}
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
