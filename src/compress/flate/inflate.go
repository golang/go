// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package flate implements the DEFLATE compressed data format, described in
// RFC 1951.  The [compress/gzip] and [compress/zlib] packages implement access
// to DEFLATE-based file formats.
package flate

import (
	"bufio"
	"io"
	"math/bits"
	"strconv"
	"sync"
)

const (
	maxCodeLen     = 16 // max length of Huffman code
	maxCodeLenMask = 15 // mask for max length of Huffman code
	// The next three numbers come from the RFC section 3.2.7, with the
	// additional proviso in section 3.2.5 which implies that distance codes
	// 30 and 31 should never occur in compressed data.
	maxNumLit  = 286
	maxNumDist = 30
	numCodes   = 19 // number of codes in Huffman meta-code
)

// Initialize the fixedHuffmanDecoder only once upon first use.
var fixedOnce sync.Once
var fixedHuffmanDecoder huffmanDecoder

// A CorruptInputError reports the presence of corrupt input at a given offset.
type CorruptInputError int64

func (e CorruptInputError) Error() string {
	return "flate: corrupt input before offset " + strconv.FormatInt(int64(e), 10)
}

// An InternalError reports an error in the flate code itself.
type InternalError string

func (e InternalError) Error() string { return "flate: internal error: " + string(e) }

// A ReadError reports an error encountered while reading input.
//
// Deprecated: No longer returned.
type ReadError struct {
	Offset int64 // byte offset where error occurred
	Err    error // error returned by underlying Read
}

func (e *ReadError) Error() string {
	return "flate: read error at offset " + strconv.FormatInt(e.Offset, 10) + ": " + e.Err.Error()
}

// A WriteError reports an error encountered while writing output.
//
// Deprecated: No longer returned.
type WriteError struct {
	Offset int64 // byte offset where error occurred
	Err    error // error returned by underlying Write
}

func (e *WriteError) Error() string {
	return "flate: write error at offset " + strconv.FormatInt(e.Offset, 10) + ": " + e.Err.Error()
}

// Resetter resets a ReadCloser returned by [NewReader] or [NewReaderDict]
// to switch to a new underlying [Reader]. This permits reusing a ReadCloser
// instead of allocating a new one.
type Resetter interface {
	// Reset discards any buffered data and resets the Resetter as if it was
	// newly initialized with the given reader.
	Reset(r io.Reader, dict []byte) error
}

// The data structure for decoding Huffman tables is based on that of
// zlib. There is a lookup table of a fixed bit width (huffmanChunkBits),
// For codes smaller than the table width, there are multiple entries
// (each combination of trailing bits has the same value). For codes
// larger than the table width, the table contains a link to an overflow
// table. The width of each entry in the link table is the maximum code
// size minus the chunk width.
//
// Note that you can do a lookup in the table even without all bits
// filled. Since the extra bits are zero, and the DEFLATE Huffman codes
// have the property that shorter codes come before longer ones, the
// bit length estimate in the result is a lower bound on the actual
// number of bits.
//
// See the following:
//	https://github.com/madler/zlib/raw/master/doc/algorithm.txt

// chunk & 15 is number of bits
// chunk >> 4 is value, including table link

const (
	huffmanChunkBits  = 9
	huffmanNumChunks  = 1 << huffmanChunkBits
	huffmanCountMask  = 15
	huffmanValueShift = 4
)

type huffmanDecoder struct {
	maxRead  int                       // the maximum number of bits we can read and not overread
	chunks   *[huffmanNumChunks]uint16 // chunks as described above
	links    [][]uint16                // overflow links
	linkMask uint32                    // mask the width of the link table
}

// Initialize Huffman decoding tables from array of code lengths.
// Following this function, h is guaranteed to be initialized into a complete
// tree (i.e., neither over-subscribed nor under-subscribed). The exception is a
// degenerate case where the tree has only a single symbol with length 1. Empty
// trees are permitted.
func (h *huffmanDecoder) init(lengths []int) bool {
	// Sanity enables additional runtime tests during Huffman
	// table construction. It's intended to be used during
	// development to supplement the currently ad-hoc unit tests.
	const sanity = false

	if h.chunks == nil {
		h.chunks = &[huffmanNumChunks]uint16{}
	}
	if h.maxRead != 0 {
		*h = huffmanDecoder{chunks: h.chunks, links: h.links}
	}

	// Count number of codes of each length,
	// compute maxRead and max length.
	var count [maxCodeLen]int
	var min, max int
	for _, n := range lengths {
		if n == 0 {
			continue
		}
		if min == 0 || n < min {
			min = n
		}
		if n > max {
			max = n
		}
		count[n&maxCodeLenMask]++
	}

	// Empty tree. The decompressor.huffSym function will fail later if the tree
	// is used. Technically, an empty tree is only valid for the HDIST tree and
	// not the HCLEN and HLIT tree. However, a stream with an empty HCLEN tree
	// is guaranteed to fail since it will attempt to use the tree to decode the
	// codes for the HLIT and HDIST trees. Similarly, an empty HLIT tree is
	// guaranteed to fail later since the compressed data section must be
	// composed of at least one symbol (the end-of-block marker).
	if max == 0 {
		return true
	}

	code := 0
	var nextcode [maxCodeLen]int
	for i := min; i <= max; i++ {
		code <<= 1
		nextcode[i&maxCodeLenMask] = code
		code += count[i&maxCodeLenMask]
	}

	// Check that the coding is complete (i.e., that we've
	// assigned all 2-to-the-max possible bit sequences).
	// Exception: To be compatible with zlib, we also need to
	// accept degenerate single-code codings. See also
	// TestDegenerateHuffmanCoding.
	if code != 1<<uint(max) && !(code == 1 && max == 1) {
		return false
	}

	h.maxRead = min
	chunks := h.chunks[:]
	for i := range chunks {
		chunks[i] = 0
	}

	if max > huffmanChunkBits {
		numLinks := 1 << (uint(max) - huffmanChunkBits)
		h.linkMask = uint32(numLinks - 1)

		// create link tables
		link := nextcode[huffmanChunkBits+1] >> 1
		if cap(h.links) < huffmanNumChunks-link {
			h.links = make([][]uint16, huffmanNumChunks-link)
		} else {
			h.links = h.links[:huffmanNumChunks-link]
		}
		for j := uint(link); j < huffmanNumChunks; j++ {
			reverse := int(bits.Reverse16(uint16(j)))
			reverse >>= uint(16 - huffmanChunkBits)
			off := j - uint(link)
			if sanity && h.chunks[reverse] != 0 {
				panic("impossible: overwriting existing chunk")
			}
			h.chunks[reverse] = uint16(off<<huffmanValueShift | (huffmanChunkBits + 1))
			if cap(h.links[off]) < numLinks {
				h.links[off] = make([]uint16, numLinks)
			} else {
				links := h.links[off][:0]
				h.links[off] = links[:numLinks]
			}
		}
	} else {
		h.links = h.links[:0]
	}

	for i, n := range lengths {
		if n == 0 {
			continue
		}
		code := nextcode[n]
		nextcode[n]++
		chunk := uint16(i<<huffmanValueShift | n)
		reverse := int(bits.Reverse16(uint16(code)))
		reverse >>= uint(16 - n)
		if n <= huffmanChunkBits {
			for off := reverse; off < len(h.chunks); off += 1 << uint(n) {
				// We should never need to overwrite
				// an existing chunk. Also, 0 is
				// never a valid chunk, because the
				// lower 4 "count" bits should be
				// between 1 and 15.
				if sanity && h.chunks[off] != 0 {
					panic("impossible: overwriting existing chunk")
				}
				h.chunks[off] = chunk
			}
		} else {
			j := reverse & (huffmanNumChunks - 1)
			if sanity && h.chunks[j]&huffmanCountMask != huffmanChunkBits+1 {
				// Longer codes should have been
				// associated with a link table above.
				panic("impossible: not an indirect chunk")
			}
			value := h.chunks[j] >> huffmanValueShift
			linktab := h.links[value]
			reverse >>= huffmanChunkBits
			for off := reverse; off < len(linktab); off += 1 << uint(n-huffmanChunkBits) {
				if sanity && linktab[off] != 0 {
					panic("impossible: overwriting existing chunk")
				}
				linktab[off] = chunk
			}
		}
	}

	if sanity {
		// Above we've sanity checked that we never overwrote
		// an existing entry. Here we additionally check that
		// we filled the tables completely.
		for i, chunk := range h.chunks {
			if chunk == 0 {
				// As an exception, in the degenerate
				// single-code case, we allow odd
				// chunks to be missing.
				if code == 1 && i%2 == 1 {
					continue
				}
				panic("impossible: missing chunk")
			}
		}
		for _, linktab := range h.links {
			for _, chunk := range linktab {
				if chunk == 0 {
					panic("impossible: missing chunk")
				}
			}
		}
	}

	return true
}

// The actual read interface needed by [NewReader].
// If the passed in [io.Reader] does not also have ReadByte,
// the [NewReader] will introduce its own buffering.
type Reader interface {
	io.Reader
	io.ByteReader
}

// Decompress state.
type decompressor struct {
	// Input source.
	r       Reader
	rBuf    *bufio.Reader // created if provided io.Reader does not implement io.ByteReader
	roffset int64

	// Huffman decoders for literal/length, distance.
	h1, h2 huffmanDecoder

	// Length arrays used to define Huffman codes.
	bits     *[maxNumLit + maxNumDist]int
	codebits *[numCodes]int

	// Output history, buffer.
	dict dictDecoder

	// Next step in the decompression,
	// and decompression state.
	step      func(*decompressor)
	stepState int
	err       error
	toRead    []byte
	hl, hd    *huffmanDecoder
	copyLen   int
	copyDist  int

	// Temporary buffer (avoids repeated allocation).
	buf [4]byte

	// Input bits, in top of b.
	b uint32

	nb    uint
	final bool
}

func (f *decompressor) nextBlock() {
	for f.nb < 1+2 {
		if f.err = f.moreBits(); f.err != nil {
			return
		}
	}
	f.final = f.b&1 == 1
	f.b >>= 1
	typ := f.b & 3
	f.b >>= 2
	f.nb -= 1 + 2
	switch typ {
	case 0:
		f.dataBlock()
	case 1:
		// compressed, fixed Huffman tables
		f.hl = &fixedHuffmanDecoder
		f.hd = nil
		f.huffmanBlockDecoder()()
	case 2:
		// compressed, dynamic Huffman tables
		if f.err = f.readHuffman(); f.err != nil {
			break
		}
		f.hl = &f.h1
		f.hd = &f.h2
		f.huffmanBlockDecoder()()
	default:
		// 3 is reserved.
		f.err = CorruptInputError(f.roffset)
	}
}

func (f *decompressor) Read(b []byte) (int, error) {
	for {
		if len(f.toRead) > 0 {
			n := copy(b, f.toRead)
			f.toRead = f.toRead[n:]
			if len(f.toRead) == 0 {
				return n, f.err
			}
			return n, nil
		}
		if f.err != nil {
			return 0, f.err
		}
		f.step(f)
		if f.err != nil && len(f.toRead) == 0 {
			f.toRead = f.dict.readFlush() // Flush what's left in case of error
		}
	}
}

func (f *decompressor) Close() error {
	if f.err == io.EOF {
		return nil
	}
	return f.err
}

// RFC 1951 section 3.2.7.
// Compression with dynamic Huffman codes

var codeOrder = [...]int{16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15}

func (f *decompressor) readHuffman() error {
	// HLIT[5], HDIST[5], HCLEN[4].
	for f.nb < 5+5+4 {
		if err := f.moreBits(); err != nil {
			return err
		}
	}
	nlit := int(f.b&0x1F) + 257
	if nlit > maxNumLit {
		return CorruptInputError(f.roffset)
	}
	f.b >>= 5
	ndist := int(f.b&0x1F) + 1
	if ndist > maxNumDist {
		return CorruptInputError(f.roffset)
	}
	f.b >>= 5
	nclen := int(f.b&0xF) + 4
	// numCodes is 19, so nclen is always valid.
	f.b >>= 4
	f.nb -= 5 + 5 + 4

	// (HCLEN+4)*3 bits: code lengths in the magic codeOrder order.
	for i := 0; i < nclen; i++ {
		for f.nb < 3 {
			if err := f.moreBits(); err != nil {
				return err
			}
		}
		f.codebits[codeOrder[i]] = int(f.b & 0x7)
		f.b >>= 3
		f.nb -= 3
	}
	for i := nclen; i < len(codeOrder); i++ {
		f.codebits[codeOrder[i]] = 0
	}
	if !f.h1.init(f.codebits[0:]) {
		return CorruptInputError(f.roffset)
	}

	// HLIT + 257 code lengths, HDIST + 1 code lengths,
	// using the code length Huffman code.
	for i, n := 0, nlit+ndist; i < n; {
		x, err := f.huffSym(&f.h1)
		if err != nil {
			return err
		}
		if x < 16 {
			// Actual length.
			f.bits[i] = x
			i++
			continue
		}
		// Repeat previous length or zero.
		var rep int
		var nb uint
		var b int
		switch x {
		default:
			return InternalError("unexpected length code")
		case 16:
			rep = 3
			nb = 2
			if i == 0 {
				return CorruptInputError(f.roffset)
			}
			b = f.bits[i-1]
		case 17:
			rep = 3
			nb = 3
			b = 0
		case 18:
			rep = 11
			nb = 7
			b = 0
		}
		for f.nb < nb {
			if err := f.moreBits(); err != nil {
				return err
			}
		}
		rep += int(f.b & uint32(1<<nb-1))
		f.b >>= nb
		f.nb -= nb
		if i+rep > n {
			return CorruptInputError(f.roffset)
		}
		for j := 0; j < rep; j++ {
			f.bits[i] = b
			i++
		}
	}

	if !f.h1.init(f.bits[0:nlit]) || !f.h2.init(f.bits[nlit:nlit+ndist]) {
		return CorruptInputError(f.roffset)
	}

	// As an optimization, we can initialize the maxRead bits to read at a time
	// for the HLIT tree to the length of the EOB marker since we know that
	// every block must terminate with one. This preserves the property that
	// we never read any extra bytes after the end of the DEFLATE stream.
	if f.h1.maxRead < f.bits[endBlockMarker] {
		f.h1.maxRead = f.bits[endBlockMarker]
	}
	if !f.final {
		// If not the final block, the smallest block possible is
		// a predefined table, BTYPE=01, with a single EOB marker.
		// This will take up 3 + 7 bits.
		f.h1.maxRead += 10
	}

	return nil
}

// Decode a single Huffman block from f.
// hl and hd are the Huffman states for the lit/length values
// and the distance values, respectively. If hd == nil, using the
// fixed distance encoding associated with fixed Huffman blocks.
func (f *decompressor) huffmanBlockGeneric() {
	const (
		stateInit = iota // Zero value must be stateInit
		stateDict
	)

	switch f.stepState {
	case stateInit:
		goto readLiteral
	case stateDict:
		goto copyHistory
	}

readLiteral:
	// Read literal and/or (length, distance) according to RFC section 3.2.3.
	{
		var v int
		{
			// Inlined v, err := f.huffSym(f.hl)
			// Since a huffmanDecoder can be empty or be composed of a degenerate tree
			// with single element, huffSym must error on these two edge cases. In both
			// cases, the chunks slice will be 0 for the invalid sequence, leading it
			// satisfy the n == 0 check below.
			n := uint(f.hl.maxRead)
			// Optimization. Compiler isn't smart enough to keep f.b,f.nb in registers,
			// but is smart enough to keep local variables in registers, so use nb and b,
			// inline call to moreBits and reassign b,nb back to f on return.
			nb, b := f.nb, f.b
			for {
				for nb < n {
					c, err := f.r.ReadByte()
					if err != nil {
						f.b = b
						f.nb = nb
						f.err = noEOF(err)
						return
					}
					f.roffset++
					b |= uint32(c) << (nb & 31)
					nb += 8
				}
				chunk := f.hl.chunks[b&(huffmanNumChunks-1)]
				n = uint(chunk & huffmanCountMask)
				if n > huffmanChunkBits {
					chunk = f.hl.links[chunk>>huffmanValueShift][(b>>huffmanChunkBits)&f.hl.linkMask]
					n = uint(chunk & huffmanCountMask)
				}
				if n <= nb {
					if n == 0 {
						f.b = b
						f.nb = nb
						f.err = CorruptInputError(f.roffset)
						return
					}
					f.b = b >> (n & 31)
					f.nb = nb - n
					v = int(chunk >> huffmanValueShift)
					break
				}
			}
		}

		var n uint // number of bits extra
		var length int
		var err error
		switch {
		case v < 256:
			f.dict.writeByte(byte(v))
			if f.dict.availWrite() == 0 {
				f.toRead = f.dict.readFlush()
				f.step = (*decompressor).huffmanBlockGeneric
				f.stepState = stateInit
				return
			}
			goto readLiteral
		case v == 256:
			f.finishBlock()
			return
		// otherwise, reference to older data
		case v < 265:
			length = v - (257 - 3)
			n = 0
		case v < 269:
			length = v*2 - (265*2 - 11)
			n = 1
		case v < 273:
			length = v*4 - (269*4 - 19)
			n = 2
		case v < 277:
			length = v*8 - (273*8 - 35)
			n = 3
		case v < 281:
			length = v*16 - (277*16 - 67)
			n = 4
		case v < 285:
			length = v*32 - (281*32 - 131)
			n = 5
		case v < maxNumLit:
			length = 258
			n = 0
		default:
			f.err = CorruptInputError(f.roffset)
			return
		}
		if n > 0 {
			for f.nb < n {
				if err = f.moreBits(); err != nil {
					f.err = err
					return
				}
			}
			length += int(f.b & uint32(1<<n-1))
			f.b >>= n
			f.nb -= n
		}

		var dist int
		if f.hd == nil {
			for f.nb < 5 {
				if err = f.moreBits(); err != nil {
					f.err = err
					return
				}
			}
			dist = int(bits.Reverse8(uint8(f.b & 0x1F << 3)))
			f.b >>= 5
			f.nb -= 5
		} else {
			if dist, err = f.huffSym(f.hd); err != nil {
				f.err = err
				return
			}
		}

		switch {
		case dist < 4:
			dist++
		case dist < maxNumDist:
			nb := uint(dist-2) >> 1
			// have 1 bit in bottom of dist, need nb more.
			extra := (dist & 1) << nb
			for f.nb < nb {
				if err = f.moreBits(); err != nil {
					f.err = err
					return
				}
			}
			extra |= int(f.b & uint32(1<<nb-1))
			f.b >>= nb
			f.nb -= nb
			dist = 1<<(nb+1) + 1 + extra
		default:
			f.err = CorruptInputError(f.roffset)
			return
		}

		// No check on length; encoding can be prescient.
		if dist > f.dict.histSize() {
			f.err = CorruptInputError(f.roffset)
			return
		}

		f.copyLen, f.copyDist = length, dist
		goto copyHistory
	}

copyHistory:
	// Perform a backwards copy according to RFC section 3.2.3.
	{
		cnt := f.dict.tryWriteCopy(f.copyDist, f.copyLen)
		if cnt == 0 {
			cnt = f.dict.writeCopy(f.copyDist, f.copyLen)
		}
		f.copyLen -= cnt

		if f.dict.availWrite() == 0 || f.copyLen > 0 {
			f.toRead = f.dict.readFlush()
			f.step = (*decompressor).huffmanBlockGeneric // We need to continue this work
			f.stepState = stateDict
			return
		}
		goto readLiteral
	}
}

// Copy a single uncompressed data block from input to output.
func (f *decompressor) dataBlock() {
	// Uncompressed.
	// Discard current partial byte.
	left := (f.nb) & 7
	f.nb -= left
	f.b >>= left

	offBytes := f.nb >> 3
	// Unfilled values will be overwritten.
	f.buf[0] = uint8(f.b)
	f.buf[1] = uint8(f.b >> 8)
	f.buf[2] = uint8(f.b >> 16)
	f.buf[3] = uint8(f.b >> 24)

	f.roffset += int64(offBytes)
	f.nb, f.b = 0, 0

	// Length then ones-complement of length.
	nr, err := io.ReadFull(f.r, f.buf[offBytes:4])
	f.roffset += int64(nr)
	if err != nil {
		f.err = noEOF(err)
		return
	}
	n := uint16(f.buf[0]) | uint16(f.buf[1])<<8
	nn := uint16(f.buf[2]) | uint16(f.buf[3])<<8
	if nn != ^n {
		f.err = CorruptInputError(f.roffset)
		return
	}

	if n == 0 {
		f.toRead = f.dict.readFlush()
		f.finishBlock()
		return
	}

	f.copyLen = int(n)
	f.copyData()
}

// copyData copies f.copyLen bytes from the underlying reader into f.hist.
// It pauses for reads when f.hist is full.
func (f *decompressor) copyData() {
	buf := f.dict.writeSlice()
	if len(buf) > f.copyLen {
		buf = buf[:f.copyLen]
	}

	cnt, err := io.ReadFull(f.r, buf)
	f.roffset += int64(cnt)
	f.copyLen -= cnt
	f.dict.writeMark(cnt)
	if err != nil {
		f.err = noEOF(err)
		return
	}

	if f.dict.availWrite() == 0 || f.copyLen > 0 {
		f.toRead = f.dict.readFlush()
		f.step = (*decompressor).copyData
		return
	}
	f.finishBlock()
}

func (f *decompressor) finishBlock() {
	if f.final {
		if f.dict.availRead() > 0 {
			f.toRead = f.dict.readFlush()
		}
		f.err = io.EOF
	}
	f.step = (*decompressor).nextBlock
}

// noEOF returns err, unless err == io.EOF, in which case it returns io.ErrUnexpectedEOF.
func noEOF(e error) error {
	if e == io.EOF {
		return io.ErrUnexpectedEOF
	}
	return e
}

func (f *decompressor) moreBits() error {
	c, err := f.r.ReadByte()
	if err != nil {
		return noEOF(err)
	}
	f.roffset++
	f.b |= uint32(c) << f.nb
	f.nb += 8
	return nil
}

// Read the next Huffman-encoded symbol from f according to h.
func (f *decompressor) huffSym(h *huffmanDecoder) (int, error) {
	// Since a huffmanDecoder can be empty or be composed of a degenerate tree
	// with single element, huffSym must error on these two edge cases. In both
	// cases, the chunks slice will be 0 for the invalid sequence, leading it
	// satisfy the n == 0 check below.
	n := uint(h.maxRead)
	// Optimization. Compiler isn't smart enough to keep f.b,f.nb in registers,
	// but is smart enough to keep local variables in registers, so use nb and b,
	// inline call to moreBits and reassign b,nb back to f on return.
	nb, b := f.nb, f.b
	for {
		for nb < n {
			c, err := f.r.ReadByte()
			if err != nil {
				f.b = b
				f.nb = nb
				return 0, noEOF(err)
			}
			f.roffset++
			b |= uint32(c) << (nb & 31)
			nb += 8
		}
		chunk := h.chunks[b&(huffmanNumChunks-1)]
		n = uint(chunk & huffmanCountMask)
		if n > huffmanChunkBits {
			chunk = h.links[chunk>>huffmanValueShift][(b>>huffmanChunkBits)&h.linkMask]
			n = uint(chunk & huffmanCountMask)
		}
		if n <= nb {
			if n == 0 {
				f.b = b
				f.nb = nb
				f.err = CorruptInputError(f.roffset)
				return 0, f.err
			}
			f.b = b >> (n & 31)
			f.nb = nb - n
			return int(chunk >> huffmanValueShift), nil
		}
	}
}

func (f *decompressor) makeReader(r io.Reader) {
	if rr, ok := r.(Reader); ok {
		f.rBuf = nil
		f.r = rr
		return
	}
	// Reuse rBuf if possible. Invariant: rBuf is always created (and owned) by decompressor.
	if f.rBuf != nil {
		f.rBuf.Reset(r)
	} else {
		// bufio.NewReader will not return r, as r does not implement flate.Reader, so it is not bufio.Reader.
		f.rBuf = bufio.NewReader(r)
	}
	f.r = f.rBuf
}

func fixedHuffmanDecoderInit() {
	fixedOnce.Do(func() {
		// These come from the RFC section 3.2.6.
		var bits [288]int
		for i := 0; i < 144; i++ {
			bits[i] = 8
		}
		for i := 144; i < 256; i++ {
			bits[i] = 9
		}
		for i := 256; i < 280; i++ {
			bits[i] = 7
		}
		for i := 280; i < 288; i++ {
			bits[i] = 8
		}
		fixedHuffmanDecoder.init(bits[:])
	})
}

func (f *decompressor) Reset(r io.Reader, dict []byte) error {
	*f = decompressor{
		rBuf:     f.rBuf,
		bits:     f.bits,
		codebits: f.codebits,
		h1:       f.h1,
		h2:       f.h2,
		dict:     f.dict,
		step:     (*decompressor).nextBlock,
	}
	f.makeReader(r)
	f.dict.init(maxMatchOffset, dict)
	return nil
}

// NewReader returns a new ReadCloser that can be used
// to read the uncompressed version of r.
// If r does not also implement [io.ByteReader],
// the decompressor may read more data than necessary from r.
// The reader returns [io.EOF] after the final block in the DEFLATE stream has
// been encountered. Any trailing data after the final block is ignored.
//
// The [io.ReadCloser] returned by NewReader also implements [Resetter].
func NewReader(r io.Reader) io.ReadCloser {
	fixedHuffmanDecoderInit()

	var f decompressor
	f.makeReader(r)
	f.bits = new([maxNumLit + maxNumDist]int)
	f.codebits = new([numCodes]int)
	f.step = (*decompressor).nextBlock
	f.dict.init(maxMatchOffset, nil)
	return &f
}

// NewReaderDict is like [NewReader] but initializes the reader
// with a preset dictionary. The returned reader behaves as if
// the uncompressed data stream started with the given dictionary,
// which has already been read. NewReaderDict is typically used
// to read data compressed by [NewWriterDict].
//
// The ReadCloser returned by NewReaderDict also implements [Resetter].
func NewReaderDict(r io.Reader, dict []byte) io.ReadCloser {
	fixedHuffmanDecoderInit()

	var f decompressor
	f.makeReader(r)
	f.bits = new([maxNumLit + maxNumDist]int)
	f.codebits = new([numCodes]int)
	f.step = (*decompressor).nextBlock
	f.dict.init(maxMatchOffset, dict)
	return &f
}
