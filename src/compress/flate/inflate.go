// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:generate go run gen.go -output fixedhuff.go

// Package flate implements the DEFLATE compressed data format, described in
// RFC 1951.  The gzip and zlib packages implement access to DEFLATE-based file
// formats.
package flate

import (
	"bufio"
	"io"
	"strconv"
)

const (
	maxCodeLen = 16    // max length of Huffman code
	maxHist    = 32768 // max history required
	// The next three numbers come from the RFC section 3.2.7, with the
	// additional proviso in section 3.2.5 which implies that distance codes
	// 30 and 31 should never occur in compressed data.
	maxNumLit  = 286
	maxNumDist = 30
	numCodes   = 19 // number of codes in Huffman meta-code
)

// A CorruptInputError reports the presence of corrupt input at a given offset.
type CorruptInputError int64

func (e CorruptInputError) Error() string {
	return "flate: corrupt input before offset " + strconv.FormatInt(int64(e), 10)
}

// An InternalError reports an error in the flate code itself.
type InternalError string

func (e InternalError) Error() string { return "flate: internal error: " + string(e) }

// A ReadError reports an error encountered while reading input.
type ReadError struct {
	Offset int64 // byte offset where error occurred
	Err    error // error returned by underlying Read
}

func (e *ReadError) Error() string {
	return "flate: read error at offset " + strconv.FormatInt(e.Offset, 10) + ": " + e.Err.Error()
}

// A WriteError reports an error encountered while writing output.
type WriteError struct {
	Offset int64 // byte offset where error occurred
	Err    error // error returned by underlying Write
}

func (e *WriteError) Error() string {
	return "flate: write error at offset " + strconv.FormatInt(e.Offset, 10) + ": " + e.Err.Error()
}

// Resetter resets a ReadCloser returned by NewReader or NewReaderDict to
// to switch to a new underlying Reader. This permits reusing a ReadCloser
// instead of allocating a new one.
type Resetter interface {
	// Reset discards any buffered data and resets the Resetter as if it was
	// newly initialized with the given reader.
	Reset(r io.Reader, dict []byte) error
}

// Note that much of the implementation of huffmanDecoder is also copied
// into gen.go (in package main) for the purpose of precomputing the
// fixed huffman tables so they can be included statically.

// The data structure for decoding Huffman tables is based on that of
// zlib. There is a lookup table of a fixed bit width (huffmanChunkBits),
// For codes smaller than the table width, there are multiple entries
// (each combination of trailing bits has the same value). For codes
// larger than the table width, the table contains a link to an overflow
// table. The width of each entry in the link table is the maximum code
// size minus the chunk width.

// Note that you can do a lookup in the table even without all bits
// filled. Since the extra bits are zero, and the DEFLATE Huffman codes
// have the property that shorter codes come before longer ones, the
// bit length estimate in the result is a lower bound on the actual
// number of bits.

// chunk & 15 is number of bits
// chunk >> 4 is value, including table link

const (
	huffmanChunkBits  = 9
	huffmanNumChunks  = 1 << huffmanChunkBits
	huffmanCountMask  = 15
	huffmanValueShift = 4
)

type huffmanDecoder struct {
	min      int                      // the minimum code length
	chunks   [huffmanNumChunks]uint32 // chunks as described above
	links    [][]uint32               // overflow links
	linkMask uint32                   // mask the width of the link table
}

// Initialize Huffman decoding tables from array of code lengths.
// Following this function, h is guaranteed to be initialized into a complete
// tree (i.e., neither over-subscribed nor under-subscribed). The exception is a
// degenerate case where the tree has only a single symbol with length 1. Empty
// trees are permitted.
func (h *huffmanDecoder) init(bits []int) bool {
	// Sanity enables additional runtime tests during Huffman
	// table construction.  It's intended to be used during
	// development to supplement the currently ad-hoc unit tests.
	const sanity = false

	if h.min != 0 {
		*h = huffmanDecoder{}
	}

	// Count number of codes of each length,
	// compute min and max length.
	var count [maxCodeLen]int
	var min, max int
	for _, n := range bits {
		if n == 0 {
			continue
		}
		if min == 0 || n < min {
			min = n
		}
		if n > max {
			max = n
		}
		count[n]++
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
		nextcode[i] = code
		code += count[i]
	}

	// Check that the coding is complete (i.e., that we've
	// assigned all 2-to-the-max possible bit sequences).
	// Exception: To be compatible with zlib, we also need to
	// accept degenerate single-code codings.  See also
	// TestDegenerateHuffmanCoding.
	if code != 1<<uint(max) && !(code == 1 && max == 1) {
		return false
	}

	h.min = min
	if max > huffmanChunkBits {
		numLinks := 1 << (uint(max) - huffmanChunkBits)
		h.linkMask = uint32(numLinks - 1)

		// create link tables
		link := nextcode[huffmanChunkBits+1] >> 1
		h.links = make([][]uint32, huffmanNumChunks-link)
		for j := uint(link); j < huffmanNumChunks; j++ {
			reverse := int(reverseByte[j>>8]) | int(reverseByte[j&0xff])<<8
			reverse >>= uint(16 - huffmanChunkBits)
			off := j - uint(link)
			if sanity && h.chunks[reverse] != 0 {
				panic("impossible: overwriting existing chunk")
			}
			h.chunks[reverse] = uint32(off<<huffmanValueShift | (huffmanChunkBits + 1))
			h.links[off] = make([]uint32, numLinks)
		}
	}

	for i, n := range bits {
		if n == 0 {
			continue
		}
		code := nextcode[n]
		nextcode[n]++
		chunk := uint32(i<<huffmanValueShift | n)
		reverse := int(reverseByte[code>>8]) | int(reverseByte[code&0xff])<<8
		reverse >>= uint(16 - n)
		if n <= huffmanChunkBits {
			for off := reverse; off < len(h.chunks); off += 1 << uint(n) {
				// We should never need to overwrite
				// an existing chunk.  Also, 0 is
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
		// an existing entry.  Here we additionally check that
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

// The actual read interface needed by NewReader.
// If the passed in io.Reader does not also have ReadByte,
// the NewReader will introduce its own buffering.
type Reader interface {
	io.Reader
	io.ByteReader
}

// Decompress state.
type decompressor struct {
	// Input source.
	r       Reader
	roffset int64
	woffset int64

	// Input bits, in top of b.
	b  uint32
	nb uint

	// Huffman decoders for literal/length, distance.
	h1, h2 huffmanDecoder

	// Length arrays used to define Huffman codes.
	bits     *[maxNumLit + maxNumDist]int
	codebits *[numCodes]int

	// Output history, buffer.
	hist  *[maxHist]byte
	hp    int  // current output position in buffer
	hw    int  // have written hist[0:hw] already
	hfull bool // buffer has filled at least once

	// Temporary buffer (avoids repeated allocation).
	buf [4]byte

	// Next step in the decompression,
	// and decompression state.
	step     func(*decompressor)
	final    bool
	err      error
	toRead   []byte
	hl, hd   *huffmanDecoder
	copyLen  int
	copyDist int
}

func (f *decompressor) nextBlock() {
	if f.final {
		if f.hw != f.hp {
			f.flush((*decompressor).nextBlock)
			return
		}
		f.err = io.EOF
		return
	}
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
		f.huffmanBlock()
	case 2:
		// compressed, dynamic Huffman tables
		if f.err = f.readHuffman(); f.err != nil {
			break
		}
		f.hl = &f.h1
		f.hd = &f.h2
		f.huffmanBlock()
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
			return n, nil
		}
		if f.err != nil {
			return 0, f.err
		}
		f.step(f)
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

	return nil
}

// Decode a single Huffman block from f.
// hl and hd are the Huffman states for the lit/length values
// and the distance values, respectively.  If hd == nil, using the
// fixed distance encoding associated with fixed Huffman blocks.
func (f *decompressor) huffmanBlock() {
	for {
		v, err := f.huffSym(f.hl)
		if err != nil {
			f.err = err
			return
		}
		var n uint // number of bits extra
		var length int
		switch {
		case v < 256:
			f.hist[f.hp] = byte(v)
			f.hp++
			if f.hp == len(f.hist) {
				// After the flush, continue this loop.
				f.flush((*decompressor).huffmanBlock)
				return
			}
			continue
		case v == 256:
			// Done with huffman block; read next block.
			f.step = (*decompressor).nextBlock
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
			dist = int(reverseByte[(f.b&0x1F)<<3])
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

		// Copy history[-dist:-dist+length] into output.
		if dist > len(f.hist) {
			f.err = InternalError("bad history distance")
			return
		}

		// No check on length; encoding can be prescient.
		if !f.hfull && dist > f.hp {
			f.err = CorruptInputError(f.roffset)
			return
		}

		f.copyLen, f.copyDist = length, dist
		if f.copyHist() {
			return
		}
	}
}

// copyHist copies f.copyLen bytes from f.hist (f.copyDist bytes ago) to itself.
// It reports whether the f.hist buffer is full.
func (f *decompressor) copyHist() bool {
	p := f.hp - f.copyDist
	if p < 0 {
		p += len(f.hist)
	}
	for f.copyLen > 0 {
		n := f.copyLen
		if x := len(f.hist) - f.hp; n > x {
			n = x
		}
		if x := len(f.hist) - p; n > x {
			n = x
		}
		forwardCopy(f.hist[:], f.hp, p, n)
		p += n
		f.hp += n
		f.copyLen -= n
		if f.hp == len(f.hist) {
			// After flush continue copying out of history.
			f.flush((*decompressor).copyHuff)
			return true
		}
		if p == len(f.hist) {
			p = 0
		}
	}
	return false
}

func (f *decompressor) copyHuff() {
	if f.copyHist() {
		return
	}
	f.huffmanBlock()
}

// Copy a single uncompressed data block from input to output.
func (f *decompressor) dataBlock() {
	// Uncompressed.
	// Discard current half-byte.
	f.nb = 0
	f.b = 0

	// Length then ones-complement of length.
	nr, err := io.ReadFull(f.r, f.buf[0:4])
	f.roffset += int64(nr)
	if err != nil {
		f.err = &ReadError{f.roffset, err}
		return
	}
	n := int(f.buf[0]) | int(f.buf[1])<<8
	nn := int(f.buf[2]) | int(f.buf[3])<<8
	if uint16(nn) != uint16(^n) {
		f.err = CorruptInputError(f.roffset)
		return
	}

	if n == 0 {
		// 0-length block means sync
		f.flush((*decompressor).nextBlock)
		return
	}

	f.copyLen = n
	f.copyData()
}

// copyData copies f.copyLen bytes from the underlying reader into f.hist.
// It pauses for reads when f.hist is full.
func (f *decompressor) copyData() {
	n := f.copyLen
	for n > 0 {
		m := len(f.hist) - f.hp
		if m > n {
			m = n
		}
		m, err := io.ReadFull(f.r, f.hist[f.hp:f.hp+m])
		f.roffset += int64(m)
		if err != nil {
			f.err = &ReadError{f.roffset, err}
			return
		}
		n -= m
		f.hp += m
		if f.hp == len(f.hist) {
			f.copyLen = n
			f.flush((*decompressor).copyData)
			return
		}
	}
	f.step = (*decompressor).nextBlock
}

func (f *decompressor) setDict(dict []byte) {
	if len(dict) > len(f.hist) {
		// Will only remember the tail.
		dict = dict[len(dict)-len(f.hist):]
	}

	f.hp = copy(f.hist[:], dict)
	if f.hp == len(f.hist) {
		f.hp = 0
		f.hfull = true
	}
	f.hw = f.hp
}

func (f *decompressor) moreBits() error {
	c, err := f.r.ReadByte()
	if err != nil {
		if err == io.EOF {
			err = io.ErrUnexpectedEOF
		}
		return err
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
	n := uint(h.min)
	for {
		for f.nb < n {
			if err := f.moreBits(); err != nil {
				return 0, err
			}
		}
		chunk := h.chunks[f.b&(huffmanNumChunks-1)]
		n = uint(chunk & huffmanCountMask)
		if n > huffmanChunkBits {
			chunk = h.links[chunk>>huffmanValueShift][(f.b>>huffmanChunkBits)&h.linkMask]
			n = uint(chunk & huffmanCountMask)
		}
		if n <= f.nb {
			if n == 0 {
				f.err = CorruptInputError(f.roffset)
				return 0, f.err
			}
			f.b >>= n
			f.nb -= n
			return int(chunk >> huffmanValueShift), nil
		}
	}
}

// Flush any buffered output to the underlying writer.
func (f *decompressor) flush(step func(*decompressor)) {
	f.toRead = f.hist[f.hw:f.hp]
	f.woffset += int64(f.hp - f.hw)
	f.hw = f.hp
	if f.hp == len(f.hist) {
		f.hp = 0
		f.hw = 0
		f.hfull = true
	}
	f.step = step
}

func makeReader(r io.Reader) Reader {
	if rr, ok := r.(Reader); ok {
		return rr
	}
	return bufio.NewReader(r)
}

func (f *decompressor) Reset(r io.Reader, dict []byte) error {
	*f = decompressor{
		r:        makeReader(r),
		bits:     f.bits,
		codebits: f.codebits,
		hist:     f.hist,
		step:     (*decompressor).nextBlock,
	}
	if dict != nil {
		f.setDict(dict)
	}
	return nil
}

// NewReader returns a new ReadCloser that can be used
// to read the uncompressed version of r.
// If r does not also implement io.ByteReader,
// the decompressor may read more data than necessary from r.
// It is the caller's responsibility to call Close on the ReadCloser
// when finished reading.
//
// The ReadCloser returned by NewReader also implements Resetter.
func NewReader(r io.Reader) io.ReadCloser {
	var f decompressor
	f.bits = new([maxNumLit + maxNumDist]int)
	f.codebits = new([numCodes]int)
	f.r = makeReader(r)
	f.hist = new([maxHist]byte)
	f.step = (*decompressor).nextBlock
	return &f
}

// NewReaderDict is like NewReader but initializes the reader
// with a preset dictionary.  The returned Reader behaves as if
// the uncompressed data stream started with the given dictionary,
// which has already been read.  NewReaderDict is typically used
// to read data compressed by NewWriterDict.
//
// The ReadCloser returned by NewReader also implements Resetter.
func NewReaderDict(r io.Reader, dict []byte) io.ReadCloser {
	var f decompressor
	f.r = makeReader(r)
	f.hist = new([maxHist]byte)
	f.bits = new([maxNumLit + maxNumDist]int)
	f.codebits = new([numCodes]int)
	f.step = (*decompressor).nextBlock
	f.setDict(dict)
	return &f
}
