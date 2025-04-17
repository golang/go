// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package zstd

import (
	"math/bits"
)

// block is the data for a single compressed block.
// The data starts immediately after the 3 byte block header,
// and is Block_Size bytes long.
type block []byte

// bitReader reads a bit stream going forward.
type bitReader struct {
	r    *Reader // for error reporting
	data block   // the bits to read
	off  uint32  // current offset into data
	bits uint32  // bits ready to be returned
	cnt  uint32  // number of valid bits in the bits field
}

// makeBitReader makes a bit reader starting at off.
func (r *Reader) makeBitReader(data block, off int) bitReader {
	return bitReader{
		r:    r,
		data: data,
		off:  uint32(off),
	}
}

// moreBits is called to read more bits.
// This ensures that at least 16 bits are available.
func (br *bitReader) moreBits() error {
	for br.cnt < 16 {
		if br.off >= uint32(len(br.data)) {
			return br.r.makeEOFError(int(br.off))
		}
		c := br.data[br.off]
		br.off++
		br.bits |= uint32(c) << br.cnt
		br.cnt += 8
	}
	return nil
}

// val is called to fetch a value of b bits.
func (br *bitReader) val(b uint8) uint32 {
	r := br.bits & ((1 << b) - 1)
	br.bits >>= b
	br.cnt -= uint32(b)
	return r
}

// backup steps back to the last byte we used.
func (br *bitReader) backup() {
	for br.cnt >= 8 {
		br.off--
		br.cnt -= 8
	}
}

// makeError returns an error at the current offset wrapping a string.
func (br *bitReader) makeError(msg string) error {
	return br.r.makeError(int(br.off), msg)
}

// reverseBitReader reads a bit stream in reverse.
type reverseBitReader struct {
	r     *Reader // for error reporting
	data  block   // the bits to read
	off   uint32  // current offset into data
	start uint32  // start in data; we read backward to start
	bits  uint32  // bits ready to be returned
	cnt   uint32  // number of valid bits in bits field
}

// makeReverseBitReader makes a reverseBitReader reading backward
// from off to start. The bitstream starts with a 1 bit in the last
// byte, at off.
func (r *Reader) makeReverseBitReader(data block, off, start int) (reverseBitReader, error) {
	streamStart := data[off]
	if streamStart == 0 {
		return reverseBitReader{}, r.makeError(off, "zero byte at reverse bit stream start")
	}
	rbr := reverseBitReader{
		r:     r,
		data:  data,
		off:   uint32(off),
		start: uint32(start),
		bits:  uint32(streamStart),
		cnt:   uint32(7 - bits.LeadingZeros8(streamStart)),
	}
	return rbr, nil
}

// val is called to fetch a value of b bits.
func (rbr *reverseBitReader) val(b uint8) (uint32, error) {
	if !rbr.fetch(b) {
		return 0, rbr.r.makeEOFError(int(rbr.off))
	}

	rbr.cnt -= uint32(b)
	v := (rbr.bits >> rbr.cnt) & ((1 << b) - 1)
	return v, nil
}

// fetch is called to ensure that at least b bits are available.
// It reports false if this can't be done,
// in which case only rbr.cnt bits are available.
func (rbr *reverseBitReader) fetch(b uint8) bool {
	for rbr.cnt < uint32(b) {
		if rbr.off <= rbr.start {
			return false
		}
		rbr.off--
		c := rbr.data[rbr.off]
		rbr.bits <<= 8
		rbr.bits |= uint32(c)
		rbr.cnt += 8
	}
	return true
}

// makeError returns an error at the current offset wrapping a string.
func (rbr *reverseBitReader) makeError(msg string) error {
	return rbr.r.makeError(int(rbr.off), msg)
}
