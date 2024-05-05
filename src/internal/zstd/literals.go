// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package zstd

import "encoding/binary"

// readLiterals reads and decompresses the literals from data at off.
// The literals are appended to outbuf, which is returned.
// Also returns the new input offset. RFC 3.1.1.3.1.
func (r *Reader) readLiterals(data block, off int, outbuf []byte) (int, []byte, error) {
	if off >= len(data) {
		return 0, nil, r.makeEOFError(off)
	}

	// Literals section header. RFC 3.1.1.3.1.1.
	hdr := data[off]
	off++

	if (hdr&3) == 0 || (hdr&3) == 1 {
		return r.readRawRLELiterals(data, off, hdr, outbuf)
	} else {
		return r.readHuffLiterals(data, off, hdr, outbuf)
	}
}

// readRawRLELiterals reads and decompresses a Raw_Literals_Block or
// a RLE_Literals_Block. RFC 3.1.1.3.1.1.
func (r *Reader) readRawRLELiterals(data block, off int, hdr byte, outbuf []byte) (int, []byte, error) {
	raw := (hdr & 3) == 0

	var regeneratedSize int
	switch (hdr >> 2) & 3 {
	case 0, 2:
		regeneratedSize = int(hdr >> 3)
	case 1:
		if off >= len(data) {
			return 0, nil, r.makeEOFError(off)
		}
		regeneratedSize = int(hdr>>4) + (int(data[off]) << 4)
		off++
	case 3:
		if off+1 >= len(data) {
			return 0, nil, r.makeEOFError(off)
		}
		regeneratedSize = int(hdr>>4) + (int(data[off]) << 4) + (int(data[off+1]) << 12)
		off += 2
	}

	// We are going to use the entire literal block in the output.
	// The maximum size of one decompressed block is 128K,
	// so we can't have more literals than that.
	if regeneratedSize > 128<<10 {
		return 0, nil, r.makeError(off, "literal size too large")
	}

	if raw {
		// RFC 3.1.1.3.1.2.
		if off+regeneratedSize > len(data) {
			return 0, nil, r.makeError(off, "raw literal size too large")
		}
		outbuf = append(outbuf, data[off:off+regeneratedSize]...)
		off += regeneratedSize
	} else {
		// RFC 3.1.1.3.1.3.
		if off >= len(data) {
			return 0, nil, r.makeError(off, "RLE literal missing")
		}
		rle := data[off]
		off++
		for i := 0; i < regeneratedSize; i++ {
			outbuf = append(outbuf, rle)
		}
	}

	return off, outbuf, nil
}

// readHuffLiterals reads and decompresses a Compressed_Literals_Block or
// a Treeless_Literals_Block. RFC 3.1.1.3.1.4.
func (r *Reader) readHuffLiterals(data block, off int, hdr byte, outbuf []byte) (int, []byte, error) {
	var (
		regeneratedSize int
		compressedSize  int
		streams         int
	)
	switch (hdr >> 2) & 3 {
	case 0, 1:
		if off+1 >= len(data) {
			return 0, nil, r.makeEOFError(off)
		}
		regeneratedSize = (int(hdr) >> 4) | ((int(data[off]) & 0x3f) << 4)
		compressedSize = (int(data[off]) >> 6) | (int(data[off+1]) << 2)
		off += 2
		if ((hdr >> 2) & 3) == 0 {
			streams = 1
		} else {
			streams = 4
		}
	case 2:
		if off+2 >= len(data) {
			return 0, nil, r.makeEOFError(off)
		}
		regeneratedSize = (int(hdr) >> 4) | (int(data[off]) << 4) | ((int(data[off+1]) & 3) << 12)
		compressedSize = (int(data[off+1]) >> 2) | (int(data[off+2]) << 6)
		off += 3
		streams = 4
	case 3:
		if off+3 >= len(data) {
			return 0, nil, r.makeEOFError(off)
		}
		regeneratedSize = (int(hdr) >> 4) | (int(data[off]) << 4) | ((int(data[off+1]) & 0x3f) << 12)
		compressedSize = (int(data[off+1]) >> 6) | (int(data[off+2]) << 2) | (int(data[off+3]) << 10)
		off += 4
		streams = 4
	}

	// We are going to use the entire literal block in the output.
	// The maximum size of one decompressed block is 128K,
	// so we can't have more literals than that.
	if regeneratedSize > 128<<10 {
		return 0, nil, r.makeError(off, "literal size too large")
	}

	roff := off + compressedSize
	if roff > len(data) || roff < 0 {
		return 0, nil, r.makeEOFError(off)
	}

	totalStreamsSize := compressedSize
	if (hdr & 3) == 2 {
		// Compressed_Literals_Block.
		// Read new huffman tree.

		if len(r.huffmanTable) < 1<<maxHuffmanBits {
			r.huffmanTable = make([]uint16, 1<<maxHuffmanBits)
		}

		huffmanTableBits, hoff, err := r.readHuff(data, off, r.huffmanTable)
		if err != nil {
			return 0, nil, err
		}
		r.huffmanTableBits = huffmanTableBits

		if totalStreamsSize < hoff-off {
			return 0, nil, r.makeError(off, "Huffman table too big")
		}
		totalStreamsSize -= hoff - off
		off = hoff
	} else {
		// Treeless_Literals_Block
		// Reuse previous Huffman tree.
		if r.huffmanTableBits == 0 {
			return 0, nil, r.makeError(off, "missing literals Huffman tree")
		}
	}

	// Decompress compressedSize bytes of data at off using the
	// Huffman tree.

	var err error
	if streams == 1 {
		outbuf, err = r.readLiteralsOneStream(data, off, totalStreamsSize, regeneratedSize, outbuf)
	} else {
		outbuf, err = r.readLiteralsFourStreams(data, off, totalStreamsSize, regeneratedSize, outbuf)
	}

	if err != nil {
		return 0, nil, err
	}

	return roff, outbuf, nil
}

// readLiteralsOneStream reads a single stream of compressed literals.
func (r *Reader) readLiteralsOneStream(data block, off, compressedSize, regeneratedSize int, outbuf []byte) ([]byte, error) {
	// We let the reverse bit reader read earlier bytes,
	// because the Huffman table ignores bits that it doesn't need.
	rbr, err := r.makeReverseBitReader(data, off+compressedSize-1, off-2)
	if err != nil {
		return nil, err
	}

	huffTable := r.huffmanTable
	huffBits := uint32(r.huffmanTableBits)
	huffMask := (uint32(1) << huffBits) - 1

	for i := 0; i < regeneratedSize; i++ {
		if !rbr.fetch(uint8(huffBits)) {
			return nil, rbr.makeError("literals Huffman stream out of bits")
		}

		var t uint16
		idx := (rbr.bits >> (rbr.cnt - huffBits)) & huffMask
		t = huffTable[idx]
		outbuf = append(outbuf, byte(t>>8))
		rbr.cnt -= uint32(t & 0xff)
	}

	return outbuf, nil
}

// readLiteralsFourStreams reads four interleaved streams of
// compressed literals.
func (r *Reader) readLiteralsFourStreams(data block, off, totalStreamsSize, regeneratedSize int, outbuf []byte) ([]byte, error) {
	// Read the jump table to find out where the streams are.
	// RFC 3.1.1.3.1.6.
	if off+5 >= len(data) {
		return nil, r.makeEOFError(off)
	}
	if totalStreamsSize < 6 {
		return nil, r.makeError(off, "total streams size too small for jump table")
	}
	// RFC 3.1.1.3.1.6.
	// "The decompressed size of each stream is equal to (Regenerated_Size+3)/4,
	// except for the last stream, which may be up to 3 bytes smaller,
	// to reach a total decompressed size as specified in Regenerated_Size."
	regeneratedStreamSize := (regeneratedSize + 3) / 4
	if regeneratedSize < regeneratedStreamSize*3 {
		return nil, r.makeError(off, "regenerated size too small to decode streams")
	}

	streamSize1 := binary.LittleEndian.Uint16(data[off:])
	streamSize2 := binary.LittleEndian.Uint16(data[off+2:])
	streamSize3 := binary.LittleEndian.Uint16(data[off+4:])
	off += 6

	tot := uint64(streamSize1) + uint64(streamSize2) + uint64(streamSize3)
	if tot > uint64(totalStreamsSize)-6 {
		return nil, r.makeEOFError(off)
	}
	streamSize4 := uint32(totalStreamsSize) - 6 - uint32(tot)

	off--
	off1 := off + int(streamSize1)
	start1 := off + 1

	off2 := off1 + int(streamSize2)
	start2 := off1 + 1

	off3 := off2 + int(streamSize3)
	start3 := off2 + 1

	off4 := off3 + int(streamSize4)
	start4 := off3 + 1

	// We let the reverse bit readers read earlier bytes,
	// because the Huffman tables ignore bits that they don't need.

	rbr1, err := r.makeReverseBitReader(data, off1, start1-2)
	if err != nil {
		return nil, err
	}

	rbr2, err := r.makeReverseBitReader(data, off2, start2-2)
	if err != nil {
		return nil, err
	}

	rbr3, err := r.makeReverseBitReader(data, off3, start3-2)
	if err != nil {
		return nil, err
	}

	rbr4, err := r.makeReverseBitReader(data, off4, start4-2)
	if err != nil {
		return nil, err
	}

	out1 := len(outbuf)
	out2 := out1 + regeneratedStreamSize
	out3 := out2 + regeneratedStreamSize
	out4 := out3 + regeneratedStreamSize

	regeneratedStreamSize4 := regeneratedSize - regeneratedStreamSize*3

	outbuf = append(outbuf, make([]byte, regeneratedSize)...)

	huffTable := r.huffmanTable
	huffBits := uint32(r.huffmanTableBits)
	huffMask := (uint32(1) << huffBits) - 1

	for i := 0; i < regeneratedStreamSize; i++ {
		use4 := i < regeneratedStreamSize4

		fetchHuff := func(rbr *reverseBitReader) (uint16, error) {
			if !rbr.fetch(uint8(huffBits)) {
				return 0, rbr.makeError("literals Huffman stream out of bits")
			}
			idx := (rbr.bits >> (rbr.cnt - huffBits)) & huffMask
			return huffTable[idx], nil
		}

		t1, err := fetchHuff(&rbr1)
		if err != nil {
			return nil, err
		}

		t2, err := fetchHuff(&rbr2)
		if err != nil {
			return nil, err
		}

		t3, err := fetchHuff(&rbr3)
		if err != nil {
			return nil, err
		}

		if use4 {
			t4, err := fetchHuff(&rbr4)
			if err != nil {
				return nil, err
			}
			outbuf[out4] = byte(t4 >> 8)
			out4++
			rbr4.cnt -= uint32(t4 & 0xff)
		}

		outbuf[out1] = byte(t1 >> 8)
		out1++
		rbr1.cnt -= uint32(t1 & 0xff)

		outbuf[out2] = byte(t2 >> 8)
		out2++
		rbr2.cnt -= uint32(t2 & 0xff)

		outbuf[out3] = byte(t3 >> 8)
		out3++
		rbr3.cnt -= uint32(t3 & 0xff)
	}

	return outbuf, nil
}
