// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package zstd

import (
	"io"
	"math/bits"
)

// maxHuffmanBits is the largest possible Huffman table bits.
const maxHuffmanBits = 11

// readHuff reads Huffman table from data starting at off into table.
// Each entry in a Huffman table is a pair of bytes.
// The high byte is the encoded value. The low byte is the number
// of bits used to encode that value. We index into the table
// with a value of size tableBits. A value that requires fewer bits
// appear in the table multiple times.
// This returns the number of bits in the Huffman table and the new offset.
// RFC 4.2.1.
func (r *Reader) readHuff(data block, off int, table []uint16) (tableBits, roff int, err error) {
	if off >= len(data) {
		return 0, 0, r.makeEOFError(off)
	}

	hdr := data[off]
	off++

	var weights [256]uint8
	var count int
	if hdr < 128 {
		// The table is compressed using an FSE. RFC 4.2.1.2.
		if len(r.fseScratch) < 1<<6 {
			r.fseScratch = make([]fseEntry, 1<<6)
		}
		fseBits, noff, err := r.readFSE(data, off, 255, 6, r.fseScratch)
		if err != nil {
			return 0, 0, err
		}
		fseTable := r.fseScratch

		if off+int(hdr) > len(data) {
			return 0, 0, r.makeEOFError(off)
		}

		rbr, err := r.makeReverseBitReader(data, off+int(hdr)-1, noff)
		if err != nil {
			return 0, 0, err
		}

		state1, err := rbr.val(uint8(fseBits))
		if err != nil {
			return 0, 0, err
		}

		state2, err := rbr.val(uint8(fseBits))
		if err != nil {
			return 0, 0, err
		}

		// There are two independent FSE streams, tracked by
		// state1 and state2. We decode them alternately.

		for {
			pt := &fseTable[state1]
			if !rbr.fetch(pt.bits) {
				if count >= 254 {
					return 0, 0, rbr.makeError("Huffman count overflow")
				}
				weights[count] = pt.sym
				weights[count+1] = fseTable[state2].sym
				count += 2
				break
			}

			v, err := rbr.val(pt.bits)
			if err != nil {
				return 0, 0, err
			}
			state1 = uint32(pt.base) + v

			if count >= 255 {
				return 0, 0, rbr.makeError("Huffman count overflow")
			}

			weights[count] = pt.sym
			count++

			pt = &fseTable[state2]

			if !rbr.fetch(pt.bits) {
				if count >= 254 {
					return 0, 0, rbr.makeError("Huffman count overflow")
				}
				weights[count] = pt.sym
				weights[count+1] = fseTable[state1].sym
				count += 2
				break
			}

			v, err = rbr.val(pt.bits)
			if err != nil {
				return 0, 0, err
			}
			state2 = uint32(pt.base) + v

			if count >= 255 {
				return 0, 0, rbr.makeError("Huffman count overflow")
			}

			weights[count] = pt.sym
			count++
		}

		off += int(hdr)
	} else {
		// The table is not compressed. Each weight is 4 bits.

		count = int(hdr) - 127
		if off+((count+1)/2) >= len(data) {
			return 0, 0, io.ErrUnexpectedEOF
		}
		for i := 0; i < count; i += 2 {
			b := data[off]
			off++
			weights[i] = b >> 4
			weights[i+1] = b & 0xf
		}
	}

	// RFC 4.2.1.3.

	var weightMark [13]uint32
	weightMask := uint32(0)
	for _, w := range weights[:count] {
		if w > 12 {
			return 0, 0, r.makeError(off, "Huffman weight overflow")
		}
		weightMark[w]++
		if w > 0 {
			weightMask += 1 << (w - 1)
		}
	}
	if weightMask == 0 {
		return 0, 0, r.makeError(off, "bad Huffman weights")
	}

	tableBits = 32 - bits.LeadingZeros32(weightMask)
	if tableBits > maxHuffmanBits {
		return 0, 0, r.makeError(off, "bad Huffman weights")
	}

	if len(table) < 1<<tableBits {
		return 0, 0, r.makeError(off, "Huffman table too small")
	}

	// Work out the last weight value, which is omitted because
	// the weights must sum to a power of two.
	left := (uint32(1) << tableBits) - weightMask
	if left == 0 {
		return 0, 0, r.makeError(off, "bad Huffman weights")
	}
	highBit := 31 - bits.LeadingZeros32(left)
	if uint32(1)<<highBit != left {
		return 0, 0, r.makeError(off, "bad Huffman weights")
	}
	if count >= 256 {
		return 0, 0, r.makeError(off, "Huffman weight overflow")
	}
	weights[count] = uint8(highBit + 1)
	count++
	weightMark[highBit+1]++

	if weightMark[1] < 2 || weightMark[1]&1 != 0 {
		return 0, 0, r.makeError(off, "bad Huffman weights")
	}

	// Change weightMark from a count of weights to the index of
	// the first symbol for that weight. We shift the indexes to
	// also store how many we have seen so far,
	next := uint32(0)
	for i := 0; i < tableBits; i++ {
		cur := next
		next += weightMark[i+1] << i
		weightMark[i+1] = cur
	}

	for i, w := range weights[:count] {
		if w == 0 {
			continue
		}
		length := uint32(1) << (w - 1)
		tval := uint16(i)<<8 | (uint16(tableBits) + 1 - uint16(w))
		start := weightMark[w]
		for j := uint32(0); j < length; j++ {
			table[start+j] = tval
		}
		weightMark[w] += length
	}

	return tableBits, off, nil
}
