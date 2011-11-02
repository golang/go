// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flate

import (
	"io"
	"math"
	"strconv"
)

const (
	// The largest offset code.
	offsetCodeCount = 30

	// The special code used to mark the end of a block.
	endBlockMarker = 256

	// The first length code.
	lengthCodesStart = 257

	// The number of codegen codes.
	codegenCodeCount = 19
	badCode          = 255
)

// The number of extra bits needed by length code X - LENGTH_CODES_START.
var lengthExtraBits = []int8{
	/* 257 */ 0, 0, 0,
	/* 260 */ 0, 0, 0, 0, 0, 1, 1, 1, 1, 2,
	/* 270 */ 2, 2, 2, 3, 3, 3, 3, 4, 4, 4,
	/* 280 */ 4, 5, 5, 5, 5, 0,
}

// The length indicated by length code X - LENGTH_CODES_START.
var lengthBase = []uint32{
	0, 1, 2, 3, 4, 5, 6, 7, 8, 10,
	12, 14, 16, 20, 24, 28, 32, 40, 48, 56,
	64, 80, 96, 112, 128, 160, 192, 224, 255,
}

// offset code word extra bits.
var offsetExtraBits = []int8{
	0, 0, 0, 0, 1, 1, 2, 2, 3, 3,
	4, 4, 5, 5, 6, 6, 7, 7, 8, 8,
	9, 9, 10, 10, 11, 11, 12, 12, 13, 13,
	/* extended window */
	14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20,
}

var offsetBase = []uint32{
	/* normal deflate */
	0x000000, 0x000001, 0x000002, 0x000003, 0x000004,
	0x000006, 0x000008, 0x00000c, 0x000010, 0x000018,
	0x000020, 0x000030, 0x000040, 0x000060, 0x000080,
	0x0000c0, 0x000100, 0x000180, 0x000200, 0x000300,
	0x000400, 0x000600, 0x000800, 0x000c00, 0x001000,
	0x001800, 0x002000, 0x003000, 0x004000, 0x006000,

	/* extended window */
	0x008000, 0x00c000, 0x010000, 0x018000, 0x020000,
	0x030000, 0x040000, 0x060000, 0x080000, 0x0c0000,
	0x100000, 0x180000, 0x200000, 0x300000,
}

// The odd order in which the codegen code sizes are written.
var codegenOrder = []uint32{16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15}

type huffmanBitWriter struct {
	w io.Writer
	// Data waiting to be written is bytes[0:nbytes]
	// and then the low nbits of bits.
	bits            uint32
	nbits           uint32
	bytes           [64]byte
	nbytes          int
	literalFreq     []int32
	offsetFreq      []int32
	codegen         []uint8
	codegenFreq     []int32
	literalEncoding *huffmanEncoder
	offsetEncoding  *huffmanEncoder
	codegenEncoding *huffmanEncoder
	err             error
}

type WrongValueError struct {
	name  string
	from  int32
	to    int32
	value int32
}

func newHuffmanBitWriter(w io.Writer) *huffmanBitWriter {
	return &huffmanBitWriter{
		w:               w,
		literalFreq:     make([]int32, maxLit),
		offsetFreq:      make([]int32, offsetCodeCount),
		codegen:         make([]uint8, maxLit+offsetCodeCount+1),
		codegenFreq:     make([]int32, codegenCodeCount),
		literalEncoding: newHuffmanEncoder(maxLit),
		offsetEncoding:  newHuffmanEncoder(offsetCodeCount),
		codegenEncoding: newHuffmanEncoder(codegenCodeCount),
	}
}

func (err WrongValueError) Error() string {
	return "huffmanBitWriter: " + err.name + " should belong to [" + strconv.Itoa64(int64(err.from)) + ";" +
		strconv.Itoa64(int64(err.to)) + "] but actual value is " + strconv.Itoa64(int64(err.value))
}

func (w *huffmanBitWriter) flushBits() {
	if w.err != nil {
		w.nbits = 0
		return
	}
	bits := w.bits
	w.bits >>= 16
	w.nbits -= 16
	n := w.nbytes
	w.bytes[n] = byte(bits)
	w.bytes[n+1] = byte(bits >> 8)
	if n += 2; n >= len(w.bytes) {
		_, w.err = w.w.Write(w.bytes[0:])
		n = 0
	}
	w.nbytes = n
}

func (w *huffmanBitWriter) flush() {
	if w.err != nil {
		w.nbits = 0
		return
	}
	n := w.nbytes
	if w.nbits > 8 {
		w.bytes[n] = byte(w.bits)
		w.bits >>= 8
		w.nbits -= 8
		n++
	}
	if w.nbits > 0 {
		w.bytes[n] = byte(w.bits)
		w.nbits = 0
		n++
	}
	w.bits = 0
	_, w.err = w.w.Write(w.bytes[0:n])
	w.nbytes = 0
}

func (w *huffmanBitWriter) writeBits(b, nb int32) {
	w.bits |= uint32(b) << w.nbits
	if w.nbits += uint32(nb); w.nbits >= 16 {
		w.flushBits()
	}
}

func (w *huffmanBitWriter) writeBytes(bytes []byte) {
	if w.err != nil {
		return
	}
	n := w.nbytes
	if w.nbits == 8 {
		w.bytes[n] = byte(w.bits)
		w.nbits = 0
		n++
	}
	if w.nbits != 0 {
		w.err = InternalError("writeBytes with unfinished bits")
		return
	}
	if n != 0 {
		_, w.err = w.w.Write(w.bytes[0:n])
		if w.err != nil {
			return
		}
	}
	w.nbytes = 0
	_, w.err = w.w.Write(bytes)
}

// RFC 1951 3.2.7 specifies a special run-length encoding for specifying
// the literal and offset lengths arrays (which are concatenated into a single
// array).  This method generates that run-length encoding.
//
// The result is written into the codegen array, and the frequencies
// of each code is written into the codegenFreq array.
// Codes 0-15 are single byte codes. Codes 16-18 are followed by additional
// information.  Code badCode is an end marker
//
//  numLiterals      The number of literals in literalEncoding
//  numOffsets       The number of offsets in offsetEncoding
func (w *huffmanBitWriter) generateCodegen(numLiterals int, numOffsets int) {
	fillInt32s(w.codegenFreq, 0)
	// Note that we are using codegen both as a temporary variable for holding
	// a copy of the frequencies, and as the place where we put the result.
	// This is fine because the output is always shorter than the input used
	// so far.
	codegen := w.codegen // cache
	// Copy the concatenated code sizes to codegen.  Put a marker at the end.
	copyUint8s(codegen[0:numLiterals], w.literalEncoding.codeBits)
	copyUint8s(codegen[numLiterals:numLiterals+numOffsets], w.offsetEncoding.codeBits)
	codegen[numLiterals+numOffsets] = badCode

	size := codegen[0]
	count := 1
	outIndex := 0
	for inIndex := 1; size != badCode; inIndex++ {
		// INVARIANT: We have seen "count" copies of size that have not yet
		// had output generated for them.
		nextSize := codegen[inIndex]
		if nextSize == size {
			count++
			continue
		}
		// We need to generate codegen indicating "count" of size.
		if size != 0 {
			codegen[outIndex] = size
			outIndex++
			w.codegenFreq[size]++
			count--
			for count >= 3 {
				n := min(count, 6)
				codegen[outIndex] = 16
				outIndex++
				codegen[outIndex] = uint8(n - 3)
				outIndex++
				w.codegenFreq[16]++
				count -= n
			}
		} else {
			for count >= 11 {
				n := min(count, 138)
				codegen[outIndex] = 18
				outIndex++
				codegen[outIndex] = uint8(n - 11)
				outIndex++
				w.codegenFreq[18]++
				count -= n
			}
			if count >= 3 {
				// count >= 3 && count <= 10
				codegen[outIndex] = 17
				outIndex++
				codegen[outIndex] = uint8(count - 3)
				outIndex++
				w.codegenFreq[17]++
				count = 0
			}
		}
		count--
		for ; count >= 0; count-- {
			codegen[outIndex] = size
			outIndex++
			w.codegenFreq[size]++
		}
		// Set up invariant for next time through the loop.
		size = nextSize
		count = 1
	}
	// Marker indicating the end of the codegen.
	codegen[outIndex] = badCode
}

func (w *huffmanBitWriter) writeCode(code *huffmanEncoder, literal uint32) {
	if w.err != nil {
		return
	}
	w.writeBits(int32(code.code[literal]), int32(code.codeBits[literal]))
}

// Write the header of a dynamic Huffman block to the output stream.
//
//  numLiterals  The number of literals specified in codegen
//  numOffsets   The number of offsets specified in codegen
//  numCodegens  The number of codegens used in codegen
func (w *huffmanBitWriter) writeDynamicHeader(numLiterals int, numOffsets int, numCodegens int, isEof bool) {
	if w.err != nil {
		return
	}
	var firstBits int32 = 4
	if isEof {
		firstBits = 5
	}
	w.writeBits(firstBits, 3)
	w.writeBits(int32(numLiterals-257), 5)
	w.writeBits(int32(numOffsets-1), 5)
	w.writeBits(int32(numCodegens-4), 4)

	for i := 0; i < numCodegens; i++ {
		value := w.codegenEncoding.codeBits[codegenOrder[i]]
		w.writeBits(int32(value), 3)
	}

	i := 0
	for {
		var codeWord int = int(w.codegen[i])
		i++
		if codeWord == badCode {
			break
		}
		// The low byte contains the actual code to generate.
		w.writeCode(w.codegenEncoding, uint32(codeWord))

		switch codeWord {
		case 16:
			w.writeBits(int32(w.codegen[i]), 2)
			i++
			break
		case 17:
			w.writeBits(int32(w.codegen[i]), 3)
			i++
			break
		case 18:
			w.writeBits(int32(w.codegen[i]), 7)
			i++
			break
		}
	}
}

func (w *huffmanBitWriter) writeStoredHeader(length int, isEof bool) {
	if w.err != nil {
		return
	}
	var flag int32
	if isEof {
		flag = 1
	}
	w.writeBits(flag, 3)
	w.flush()
	w.writeBits(int32(length), 16)
	w.writeBits(int32(^uint16(length)), 16)
}

func (w *huffmanBitWriter) writeFixedHeader(isEof bool) {
	if w.err != nil {
		return
	}
	// Indicate that we are a fixed Huffman block
	var value int32 = 2
	if isEof {
		value = 3
	}
	w.writeBits(value, 3)
}

func (w *huffmanBitWriter) writeBlock(tokens []token, eof bool, input []byte) {
	if w.err != nil {
		return
	}
	fillInt32s(w.literalFreq, 0)
	fillInt32s(w.offsetFreq, 0)

	n := len(tokens)
	tokens = tokens[0 : n+1]
	tokens[n] = endBlockMarker

	for _, t := range tokens {
		switch t.typ() {
		case literalType:
			w.literalFreq[t.literal()]++
		case matchType:
			length := t.length()
			offset := t.offset()
			w.literalFreq[lengthCodesStart+lengthCode(length)]++
			w.offsetFreq[offsetCode(offset)]++
		}
	}

	// get the number of literals
	numLiterals := len(w.literalFreq)
	for w.literalFreq[numLiterals-1] == 0 {
		numLiterals--
	}
	// get the number of offsets
	numOffsets := len(w.offsetFreq)
	for numOffsets > 0 && w.offsetFreq[numOffsets-1] == 0 {
		numOffsets--
	}
	if numOffsets == 0 {
		// We haven't found a single match. If we want to go with the dynamic encoding,
		// we should count at least one offset to be sure that the offset huffman tree could be encoded.
		w.offsetFreq[0] = 1
		numOffsets = 1
	}

	w.literalEncoding.generate(w.literalFreq, 15)
	w.offsetEncoding.generate(w.offsetFreq, 15)

	storedBytes := 0
	if input != nil {
		storedBytes = len(input)
	}
	var extraBits int64
	var storedSize int64 = math.MaxInt64
	if storedBytes <= maxStoreBlockSize && input != nil {
		storedSize = int64((storedBytes + 5) * 8)
		// We only bother calculating the costs of the extra bits required by
		// the length of offset fields (which will be the same for both fixed
		// and dynamic encoding), if we need to compare those two encodings
		// against stored encoding.
		for lengthCode := lengthCodesStart + 8; lengthCode < numLiterals; lengthCode++ {
			// First eight length codes have extra size = 0.
			extraBits += int64(w.literalFreq[lengthCode]) * int64(lengthExtraBits[lengthCode-lengthCodesStart])
		}
		for offsetCode := 4; offsetCode < numOffsets; offsetCode++ {
			// First four offset codes have extra size = 0.
			extraBits += int64(w.offsetFreq[offsetCode]) * int64(offsetExtraBits[offsetCode])
		}
	}

	// Figure out smallest code.
	// Fixed Huffman baseline.
	var size = int64(3) +
		fixedLiteralEncoding.bitLength(w.literalFreq) +
		fixedOffsetEncoding.bitLength(w.offsetFreq) +
		extraBits
	var literalEncoding = fixedLiteralEncoding
	var offsetEncoding = fixedOffsetEncoding

	// Dynamic Huffman?
	var numCodegens int

	// Generate codegen and codegenFrequencies, which indicates how to encode
	// the literalEncoding and the offsetEncoding.
	w.generateCodegen(numLiterals, numOffsets)
	w.codegenEncoding.generate(w.codegenFreq, 7)
	numCodegens = len(w.codegenFreq)
	for numCodegens > 4 && w.codegenFreq[codegenOrder[numCodegens-1]] == 0 {
		numCodegens--
	}
	dynamicHeader := int64(3+5+5+4+(3*numCodegens)) +
		w.codegenEncoding.bitLength(w.codegenFreq) +
		int64(extraBits) +
		int64(w.codegenFreq[16]*2) +
		int64(w.codegenFreq[17]*3) +
		int64(w.codegenFreq[18]*7)
	dynamicSize := dynamicHeader +
		w.literalEncoding.bitLength(w.literalFreq) +
		w.offsetEncoding.bitLength(w.offsetFreq)

	if dynamicSize < size {
		size = dynamicSize
		literalEncoding = w.literalEncoding
		offsetEncoding = w.offsetEncoding
	}

	// Stored bytes?
	if storedSize < size {
		w.writeStoredHeader(storedBytes, eof)
		w.writeBytes(input[0:storedBytes])
		return
	}

	// Huffman.
	if literalEncoding == fixedLiteralEncoding {
		w.writeFixedHeader(eof)
	} else {
		w.writeDynamicHeader(numLiterals, numOffsets, numCodegens, eof)
	}
	for _, t := range tokens {
		switch t.typ() {
		case literalType:
			w.writeCode(literalEncoding, t.literal())
			break
		case matchType:
			// Write the length
			length := t.length()
			lengthCode := lengthCode(length)
			w.writeCode(literalEncoding, lengthCode+lengthCodesStart)
			extraLengthBits := int32(lengthExtraBits[lengthCode])
			if extraLengthBits > 0 {
				extraLength := int32(length - lengthBase[lengthCode])
				w.writeBits(extraLength, extraLengthBits)
			}
			// Write the offset
			offset := t.offset()
			offsetCode := offsetCode(offset)
			w.writeCode(offsetEncoding, offsetCode)
			extraOffsetBits := int32(offsetExtraBits[offsetCode])
			if extraOffsetBits > 0 {
				extraOffset := int32(offset - offsetBase[offsetCode])
				w.writeBits(extraOffset, extraOffsetBits)
			}
			break
		default:
			panic("unknown token type: " + string(t))
		}
	}
}
