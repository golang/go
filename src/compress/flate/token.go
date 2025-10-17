// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flate

import (
	"math"
)

const (
	// Token is a compound value:
	// bits 0-16  xoffset = offset - MIN_OFFSET_SIZE, or literal - 16 bits
	// bits 16-22 offset code - 5 bits
	// bits 22-30 xlength = length - MIN_MATCH_LENGTH - 8 bits
	// bits 30-32 type, 0 = literal  1=EOF  2=Match   3=Unused - 2 bits
	lengthShift         = 22
	offsetMask          = 1<<lengthShift - 1
	typeMask            = 3 << 30
	matchType           = 1 << 30
	matchOffsetOnlyMask = 0xffff
)

// The length code for length X (MIN_MATCH_LENGTH <= X <= MAX_MATCH_LENGTH)
// is lengthCodes[length - MIN_MATCH_LENGTH]
var lengthCodes = [256]uint8{
	0, 1, 2, 3, 4, 5, 6, 7, 8, 8,
	9, 9, 10, 10, 11, 11, 12, 12, 12, 12,
	13, 13, 13, 13, 14, 14, 14, 14, 15, 15,
	15, 15, 16, 16, 16, 16, 16, 16, 16, 16,
	17, 17, 17, 17, 17, 17, 17, 17, 18, 18,
	18, 18, 18, 18, 18, 18, 19, 19, 19, 19,
	19, 19, 19, 19, 20, 20, 20, 20, 20, 20,
	20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
	21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
	21, 21, 21, 21, 21, 21, 22, 22, 22, 22,
	22, 22, 22, 22, 22, 22, 22, 22, 22, 22,
	22, 22, 23, 23, 23, 23, 23, 23, 23, 23,
	23, 23, 23, 23, 23, 23, 23, 23, 24, 24,
	24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
	24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
	24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
	25, 25, 25, 25, 25, 25, 25, 25, 25, 25,
	25, 25, 25, 25, 25, 25, 25, 25, 25, 25,
	25, 25, 25, 25, 25, 25, 25, 25, 25, 25,
	25, 25, 26, 26, 26, 26, 26, 26, 26, 26,
	26, 26, 26, 26, 26, 26, 26, 26, 26, 26,
	26, 26, 26, 26, 26, 26, 26, 26, 26, 26,
	26, 26, 26, 26, 27, 27, 27, 27, 27, 27,
	27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
	27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
	27, 27, 27, 27, 27, 28,
}

// lengthCodes1 is length codes, but starting at 1.
var lengthCodes1 = [256]uint8{
	1, 2, 3, 4, 5, 6, 7, 8, 9, 9,
	10, 10, 11, 11, 12, 12, 13, 13, 13, 13,
	14, 14, 14, 14, 15, 15, 15, 15, 16, 16,
	16, 16, 17, 17, 17, 17, 17, 17, 17, 17,
	18, 18, 18, 18, 18, 18, 18, 18, 19, 19,
	19, 19, 19, 19, 19, 19, 20, 20, 20, 20,
	20, 20, 20, 20, 21, 21, 21, 21, 21, 21,
	21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
	22, 22, 22, 22, 22, 22, 22, 22, 22, 22,
	22, 22, 22, 22, 22, 22, 23, 23, 23, 23,
	23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
	23, 23, 24, 24, 24, 24, 24, 24, 24, 24,
	24, 24, 24, 24, 24, 24, 24, 24, 25, 25,
	25, 25, 25, 25, 25, 25, 25, 25, 25, 25,
	25, 25, 25, 25, 25, 25, 25, 25, 25, 25,
	25, 25, 25, 25, 25, 25, 25, 25, 25, 25,
	26, 26, 26, 26, 26, 26, 26, 26, 26, 26,
	26, 26, 26, 26, 26, 26, 26, 26, 26, 26,
	26, 26, 26, 26, 26, 26, 26, 26, 26, 26,
	26, 26, 27, 27, 27, 27, 27, 27, 27, 27,
	27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
	27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
	27, 27, 27, 27, 28, 28, 28, 28, 28, 28,
	28, 28, 28, 28, 28, 28, 28, 28, 28, 28,
	28, 28, 28, 28, 28, 28, 28, 28, 28, 28,
	28, 28, 28, 28, 28, 29,
}

var offsetCodes = [256]uint32{
	0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7,
	8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,
	10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
	11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
	12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
	12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
	13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
	13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
	14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
	14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
	14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
	14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
	15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
	15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
	15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
	15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
}

// offsetCodes14 are offsetCodes, but with 14 added.
var offsetCodes14 = [256]uint32{
	14, 15, 16, 17, 18, 18, 19, 19, 20, 20, 20, 20, 21, 21, 21, 21,
	22, 22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23, 23, 23,
	24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
	25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25,
	26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26,
	26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26,
	27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
	27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
	28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28,
	28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28,
	28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28,
	28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28,
	29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29,
	29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29,
	29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29,
	29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29,
}

type token uint32

// tokens are compound values as described above.
// Histograms are created as tokens are added.
// A full block is allocated.
type tokens struct {
	extraHist [32]uint16  // codes 256->maxnumlit
	offHist   [32]uint16  // offset codes
	litHist   [256]uint16 // codes 0->255
	nFilled   int
	n         uint16 // Must be able to contain maxStoreBlockSize
	tokens    [65536]token
}

func (t *tokens) Reset() {
	if t.n == 0 {
		return
	}
	t.n = 0
	t.nFilled = 0
	clear(t.litHist[:])
	clear(t.extraHist[:])
	clear(t.offHist[:])
}

func indexTokens(in []token) tokens {
	var t tokens
	t.indexTokens(in)
	return t
}

func (t *tokens) indexTokens(in []token) {
	t.Reset()
	for _, tok := range in {
		if tok < matchType {
			t.AddLiteral(tok.literal())
			continue
		}
		t.AddMatch(uint32(tok.length()), tok.offset()&matchOffsetOnlyMask)
	}
}

// emitLiterals writes a literal chunk and returns the number of bytes written.
func emitLiterals(dst *tokens, lit []byte) {
	for _, v := range lit {
		dst.tokens[dst.n] = token(v)
		dst.litHist[v]++
		dst.n++
	}
}

func (t *tokens) AddLiteral(lit byte) {
	t.tokens[t.n] = token(lit)
	t.litHist[lit]++
	t.n++
}

// from https://stackoverflow.com/a/28730362
func mFastLog2(val float32) float32 {
	ux := int32(math.Float32bits(val))
	log2 := (float32)(((ux >> 23) & 255) - 128)
	ux &= -0x7f800001
	ux += 127 << 23
	uval := math.Float32frombits(uint32(ux))
	log2 += ((-0.34484843)*uval+2.02466578)*uval - 0.67487759
	return log2
}

// EstimatedBits will return a minimum size estimated by an *optimal*
// compression of the block.
func (t *tokens) EstimatedBits() int {
	shannon := float32(0)
	bits := int(0)
	nMatches := 0
	total := int(t.n) + t.nFilled
	if total > 0 {
		invTotal := 1.0 / float32(total)
		for _, v := range t.litHist[:] {
			if v > 0 {
				n := float32(v)
				shannon += atLeastOne(-mFastLog2(n*invTotal)) * n
			}
		}
		// Just add 15 for EOB
		shannon += 15
		for i, v := range t.extraHist[1 : literalCount-256] {
			if v > 0 {
				n := float32(v)
				shannon += atLeastOne(-mFastLog2(n*invTotal)) * n
				bits += int(lengthExtraBits[i&31]) * int(v)
				nMatches += int(v)
			}
		}
	}
	if nMatches > 0 {
		invTotal := 1.0 / float32(nMatches)
		for i, v := range t.offHist[:offsetCodeCount] {
			if v > 0 {
				n := float32(v)
				shannon += atLeastOne(-mFastLog2(n*invTotal)) * n
				bits += int(offsetExtraBits[i&31]) * int(v)
			}
		}
	}
	return int(shannon) + bits
}

// AddMatch adds a match to the tokens.
// This function is very sensitive to inlining and right on the border.
func (t *tokens) AddMatch(xlength uint32, xoffset uint32) {
	oCode := offsetCode(xoffset)
	xoffset |= oCode << 16

	t.extraHist[lengthCodes1[uint8(xlength)]]++
	t.offHist[oCode&31]++
	t.tokens[t.n] = token(matchType | xlength<<lengthShift | xoffset)
	t.n++
}

// AddMatchLong adds a match to the tokens, potentially longer than max match length.
// Length should NOT have the base subtracted, only offset should.
func (t *tokens) AddMatchLong(xlength int32, xoffset uint32) {
	oc := offsetCode(xoffset)
	xoffset |= oc << 16
	for xlength > 0 {
		xl := xlength
		if xl > 258 {
			// We need to have at least baseMatchLength left over for next loop.
			if xl > 258+baseMatchLength {
				xl = 258
			} else {
				xl = 258 - baseMatchLength
			}
		}
		xlength -= xl
		xl -= baseMatchLength
		t.extraHist[lengthCodes1[uint8(xl)]]++
		t.offHist[oc&31]++
		t.tokens[t.n] = token(matchType | uint32(xl)<<lengthShift | xoffset)
		t.n++
	}
}

func (t *tokens) AddEOB() {
	t.tokens[t.n] = token(endBlockMarker)
	t.extraHist[0]++
	t.n++
}

// Slice returns a slice of the tokens that references the tokens in t.
func (t *tokens) Slice() []token {
	return t.tokens[:t.n]
}

// Returns the type of a token
func (t token) typ() uint32 { return uint32(t) & typeMask }

// Returns the literal of a literal token
func (t token) literal() uint8 { return uint8(t) }

// Returns the extra offset of a match token
func (t token) offset() uint32 { return uint32(t) & offsetMask }

func (t token) length() uint8 { return uint8(t >> lengthShift) }

// Convert length to code.
func lengthCode(len uint8) uint8 { return lengthCodes[len] }

// Returns the offset code corresponding to a specific offset
func offsetCode(off uint32) uint32 {
	if off < uint32(len(offsetCodes)) {
		return offsetCodes[uint8(off)]
	}
	return offsetCodes14[uint8(off>>7)]
}
