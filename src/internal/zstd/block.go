// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package zstd

import (
	"io"
)

// debug can be set in the source to print debug info using println.
const debug = false

// compressedBlock decompresses a compressed block, storing the decompressed
// data in r.buffer. The blockSize argument is the compressed size.
// RFC 3.1.1.3.
func (r *Reader) compressedBlock(blockSize int) error {
	if len(r.compressedBuf) >= blockSize {
		r.compressedBuf = r.compressedBuf[:blockSize]
	} else {
		// We know that blockSize <= 128K,
		// so this won't allocate an enormous amount.
		need := blockSize - len(r.compressedBuf)
		r.compressedBuf = append(r.compressedBuf, make([]byte, need)...)
	}

	if _, err := io.ReadFull(r.r, r.compressedBuf); err != nil {
		return r.wrapNonEOFError(0, err)
	}

	data := block(r.compressedBuf)
	off := 0
	r.buffer = r.buffer[:0]

	litoff, litbuf, err := r.readLiterals(data, off, r.literals[:0])
	if err != nil {
		return err
	}
	r.literals = litbuf

	off = litoff

	seqCount, off, err := r.initSeqs(data, off)
	if err != nil {
		return err
	}

	if seqCount == 0 {
		// No sequences, just literals.
		if off < len(data) {
			return r.makeError(off, "extraneous data after no sequences")
		}

		r.buffer = append(r.buffer, litbuf...)

		return nil
	}

	return r.execSeqs(data, off, litbuf, seqCount)
}

// seqCode is the kind of sequence codes we have to handle.
type seqCode int

const (
	seqLiteral seqCode = iota
	seqOffset
	seqMatch
)

// seqCodeInfoData is the information needed to set up seqTables and
// seqTableBits for a particular kind of sequence code.
type seqCodeInfoData struct {
	predefTable     []fseBaselineEntry // predefined FSE
	predefTableBits int                // number of bits in predefTable
	maxSym          int                // max symbol value in FSE
	maxBits         int                // max bits for FSE

	// toBaseline converts from an FSE table to an FSE baseline table.
	toBaseline func(*Reader, int, []fseEntry, []fseBaselineEntry) error
}

// seqCodeInfo is the seqCodeInfoData for each kind of sequence code.
var seqCodeInfo = [3]seqCodeInfoData{
	seqLiteral: {
		predefTable:     predefinedLiteralTable[:],
		predefTableBits: 6,
		maxSym:          35,
		maxBits:         9,
		toBaseline:      (*Reader).makeLiteralBaselineFSE,
	},
	seqOffset: {
		predefTable:     predefinedOffsetTable[:],
		predefTableBits: 5,
		maxSym:          31,
		maxBits:         8,
		toBaseline:      (*Reader).makeOffsetBaselineFSE,
	},
	seqMatch: {
		predefTable:     predefinedMatchTable[:],
		predefTableBits: 6,
		maxSym:          52,
		maxBits:         9,
		toBaseline:      (*Reader).makeMatchBaselineFSE,
	},
}

// initSeqs reads the Sequences_Section_Header and sets up the FSE
// tables used to read the sequence codes. It returns the number of
// sequences and the new offset. RFC 3.1.1.3.2.1.
func (r *Reader) initSeqs(data block, off int) (int, int, error) {
	if off >= len(data) {
		return 0, 0, r.makeEOFError(off)
	}

	seqHdr := data[off]
	off++
	if seqHdr == 0 {
		return 0, off, nil
	}

	var seqCount int
	if seqHdr < 128 {
		seqCount = int(seqHdr)
	} else if seqHdr < 255 {
		if off >= len(data) {
			return 0, 0, r.makeEOFError(off)
		}
		seqCount = ((int(seqHdr) - 128) << 8) + int(data[off])
		off++
	} else {
		if off+1 >= len(data) {
			return 0, 0, r.makeEOFError(off)
		}
		seqCount = int(data[off]) + (int(data[off+1]) << 8) + 0x7f00
		off += 2
	}

	// Read the Symbol_Compression_Modes byte.

	if off >= len(data) {
		return 0, 0, r.makeEOFError(off)
	}
	symMode := data[off]
	if symMode&3 != 0 {
		return 0, 0, r.makeError(off, "invalid symbol compression mode")
	}
	off++

	// Set up the FSE tables used to decode the sequence codes.

	var err error
	off, err = r.setSeqTable(data, off, seqLiteral, (symMode>>6)&3)
	if err != nil {
		return 0, 0, err
	}

	off, err = r.setSeqTable(data, off, seqOffset, (symMode>>4)&3)
	if err != nil {
		return 0, 0, err
	}

	off, err = r.setSeqTable(data, off, seqMatch, (symMode>>2)&3)
	if err != nil {
		return 0, 0, err
	}

	return seqCount, off, nil
}

// setSeqTable uses the Compression_Mode in mode to set up r.seqTables and
// r.seqTableBits for kind. We store these in the Reader because one of
// the modes simply reuses the value from the last block in the frame.
func (r *Reader) setSeqTable(data block, off int, kind seqCode, mode byte) (int, error) {
	info := &seqCodeInfo[kind]
	switch mode {
	case 0:
		// Predefined_Mode
		r.seqTables[kind] = info.predefTable
		r.seqTableBits[kind] = uint8(info.predefTableBits)
		return off, nil

	case 1:
		// RLE_Mode
		if off >= len(data) {
			return 0, r.makeEOFError(off)
		}
		rle := data[off]
		off++

		// Build a simple baseline table that always returns rle.

		entry := []fseEntry{
			{
				sym:  rle,
				bits: 0,
				base: 0,
			},
		}
		if cap(r.seqTableBuffers[kind]) == 0 {
			r.seqTableBuffers[kind] = make([]fseBaselineEntry, 1<<info.maxBits)
		}
		r.seqTableBuffers[kind] = r.seqTableBuffers[kind][:1]
		if err := info.toBaseline(r, off, entry, r.seqTableBuffers[kind]); err != nil {
			return 0, err
		}

		r.seqTables[kind] = r.seqTableBuffers[kind]
		r.seqTableBits[kind] = 0
		return off, nil

	case 2:
		// FSE_Compressed_Mode
		if cap(r.fseScratch) < 1<<info.maxBits {
			r.fseScratch = make([]fseEntry, 1<<info.maxBits)
		}
		r.fseScratch = r.fseScratch[:1<<info.maxBits]

		tableBits, roff, err := r.readFSE(data, off, info.maxSym, info.maxBits, r.fseScratch)
		if err != nil {
			return 0, err
		}
		r.fseScratch = r.fseScratch[:1<<tableBits]

		if cap(r.seqTableBuffers[kind]) == 0 {
			r.seqTableBuffers[kind] = make([]fseBaselineEntry, 1<<info.maxBits)
		}
		r.seqTableBuffers[kind] = r.seqTableBuffers[kind][:1<<tableBits]

		if err := info.toBaseline(r, roff, r.fseScratch, r.seqTableBuffers[kind]); err != nil {
			return 0, err
		}

		r.seqTables[kind] = r.seqTableBuffers[kind]
		r.seqTableBits[kind] = uint8(tableBits)
		return roff, nil

	case 3:
		// Repeat_Mode
		if len(r.seqTables[kind]) == 0 {
			return 0, r.makeError(off, "missing repeat sequence FSE table")
		}
		return off, nil
	}
	panic("unreachable")
}

// execSeqs reads and executes the sequences. RFC 3.1.1.3.2.1.2.
func (r *Reader) execSeqs(data block, off int, litbuf []byte, seqCount int) error {
	// Set up the initial states for the sequence code readers.

	rbr, err := r.makeReverseBitReader(data, len(data)-1, off)
	if err != nil {
		return err
	}

	literalState, err := rbr.val(r.seqTableBits[seqLiteral])
	if err != nil {
		return err
	}

	offsetState, err := rbr.val(r.seqTableBits[seqOffset])
	if err != nil {
		return err
	}

	matchState, err := rbr.val(r.seqTableBits[seqMatch])
	if err != nil {
		return err
	}

	// Read and perform all the sequences. RFC 3.1.1.4.

	seq := 0
	for seq < seqCount {
		ptoffset := &r.seqTables[seqOffset][offsetState]
		ptmatch := &r.seqTables[seqMatch][matchState]
		ptliteral := &r.seqTables[seqLiteral][literalState]

		add, err := rbr.val(ptoffset.basebits)
		if err != nil {
			return err
		}
		offset := ptoffset.baseline + add

		add, err = rbr.val(ptmatch.basebits)
		if err != nil {
			return err
		}
		match := ptmatch.baseline + add

		add, err = rbr.val(ptliteral.basebits)
		if err != nil {
			return err
		}
		literal := ptliteral.baseline + add

		// Handle repeat offsets. RFC 3.1.1.5.
		// See the comment in makeOffsetBaselineFSE.
		if ptoffset.basebits > 1 {
			r.repeatedOffset3 = r.repeatedOffset2
			r.repeatedOffset2 = r.repeatedOffset1
			r.repeatedOffset1 = offset
		} else {
			if literal == 0 {
				offset++
			}
			switch offset {
			case 1:
				offset = r.repeatedOffset1
			case 2:
				offset = r.repeatedOffset2
				r.repeatedOffset2 = r.repeatedOffset1
				r.repeatedOffset1 = offset
			case 3:
				offset = r.repeatedOffset3
				r.repeatedOffset3 = r.repeatedOffset2
				r.repeatedOffset2 = r.repeatedOffset1
				r.repeatedOffset1 = offset
			case 4:
				offset = r.repeatedOffset1 - 1
				r.repeatedOffset3 = r.repeatedOffset2
				r.repeatedOffset2 = r.repeatedOffset1
				r.repeatedOffset1 = offset
			}
		}

		seq++
		if seq < seqCount {
			// Update the states.
			add, err = rbr.val(ptliteral.bits)
			if err != nil {
				return err
			}
			literalState = uint32(ptliteral.base) + add

			add, err = rbr.val(ptmatch.bits)
			if err != nil {
				return err
			}
			matchState = uint32(ptmatch.base) + add

			add, err = rbr.val(ptoffset.bits)
			if err != nil {
				return err
			}
			offsetState = uint32(ptoffset.base) + add
		}

		// The next sequence is now in literal, offset, match.

		if debug {
			println("literal", literal, "offset", offset, "match", match)
		}

		// Copy literal bytes from litbuf.
		if literal > uint32(len(litbuf)) {
			return rbr.makeError("literal byte overflow")
		}

		// RFC 3.1.1.2.4
		// "Block_Maximum_Size is constant for a given frame.
		// This maximum is applicable to both the decompressed size
		// and the compressed size of any block in the frame."
		if int(literal+match) > r.blockMaximumSize {
			return rbr.makeError("uncompressed size too big")
		}

		if literal > 0 {
			r.buffer = append(r.buffer, litbuf[:literal]...)
			litbuf = litbuf[literal:]
		}

		if match > 0 {
			if err := r.copyFromWindow(&rbr, offset, match); err != nil {
				return err
			}
		}
	}

	r.buffer = append(r.buffer, litbuf...)

	if rbr.cnt != 0 {
		return r.makeError(off, "extraneous data after sequences")
	}

	return nil
}

// Copy match bytes from the decoded output, or the window, at offset.
func (r *Reader) copyFromWindow(rbr *reverseBitReader, offset, match uint32) error {
	if offset == 0 {
		return rbr.makeError("invalid zero offset")
	}

	// Offset may point into the buffer or the window and
	// match may extend past the end of the initial buffer.
	// |--r.window--|--r.buffer--|
	//        |<-----offset------|
	//        |------match----------->|
	bufferOffset := uint32(0)
	lenBlock := uint32(len(r.buffer))
	if lenBlock < offset {
		lenWindow := r.window.len()
		copy := offset - lenBlock
		if copy > lenWindow {
			return rbr.makeError("offset past window")
		}
		windowOffset := lenWindow - copy
		if copy > match {
			copy = match
		}
		r.buffer = r.window.appendTo(r.buffer, windowOffset, windowOffset+copy)
		match -= copy
	} else {
		bufferOffset = lenBlock - offset
	}

	// We are being asked to copy data that we are adding to the
	// buffer in the same copy.
	for match > 0 {
		copy := uint32(len(r.buffer)) - bufferOffset
		if copy > match {
			copy = match
		}
		r.buffer = append(r.buffer, r.buffer[bufferOffset:bufferOffset+copy]...)
		match -= copy
	}
	return nil
}
