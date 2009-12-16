// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package jpeg

import (
	"io"
	"os"
)

// Each code is at most 16 bits long.
const maxCodeLength = 16

// Each decoded value is a uint8, so there are at most 256 such values.
const maxNumValues = 256

// Bit stream for the Huffman decoder.
// The n least significant bits of a form the unread bits, to be read in MSB to LSB order.
type bits struct {
	a int // accumulator.
	n int // the number of unread bits in a.
	m int // mask. m==1<<(n-1) when n>0, with m==0 when n==0.
}

// Huffman table decoder, specified in section C.
type huffman struct {
	l        [maxCodeLength]int
	length   int                 // sum of l[i].
	val      [maxNumValues]uint8 // the decoded values, as sorted by their encoding.
	size     [maxNumValues]int   // size[i] is the number of bits to encode val[i].
	code     [maxNumValues]int   // code[i] is the encoding of val[i].
	minCode  [maxCodeLength]int  // min codes of length i, or -1 if no codes of that length.
	maxCode  [maxCodeLength]int  // max codes of length i, or -1 if no codes of that length.
	valIndex [maxCodeLength]int  // index into val of minCode[i].
}

// Reads bytes from the io.Reader to ensure that bits.n is at least n.
func (d *decoder) ensureNBits(n int) os.Error {
	for d.b.n < n {
		c, err := d.r.ReadByte()
		if err != nil {
			return err
		}
		d.b.a = d.b.a<<8 | int(c)
		d.b.n += 8
		if d.b.m == 0 {
			d.b.m = 1 << 7
		} else {
			d.b.m <<= 8
		}
		// Byte stuffing, specified in section F.1.2.3.
		if c == 0xff {
			c, err = d.r.ReadByte()
			if err != nil {
				return err
			}
			if c != 0x00 {
				return FormatError("missing 0xff00 sequence")
			}
		}
	}
	return nil
}

// The composition of RECEIVE and EXTEND, specified in section F.2.2.1.
func (d *decoder) receiveExtend(t uint8) (int, os.Error) {
	err := d.ensureNBits(int(t))
	if err != nil {
		return 0, err
	}
	d.b.n -= int(t)
	d.b.m >>= t
	s := 1 << t
	x := (d.b.a >> uint8(d.b.n)) & (s - 1)
	if x < s>>1 {
		x += ((-1) << t) + 1
	}
	return x, nil
}

// Processes a Define Huffman Table marker, and initializes a huffman struct from its contents.
// Specified in section B.2.4.2.
func (d *decoder) processDHT(n int) os.Error {
	for n > 0 {
		if n < 17 {
			return FormatError("DHT has wrong length")
		}
		_, err := io.ReadFull(d.r, d.tmp[0:17])
		if err != nil {
			return err
		}
		tc := d.tmp[0] >> 4
		if tc > maxTc {
			return FormatError("bad Tc value")
		}
		th := d.tmp[0] & 0x0f
		const isBaseline = true // Progressive mode is not yet supported.
		if th > maxTh || isBaseline && th > 1 {
			return FormatError("bad Th value")
		}
		h := &d.huff[tc][th]

		// Read l and val (and derive length).
		h.length = 0
		for i := 0; i < maxCodeLength; i++ {
			h.l[i] = int(d.tmp[i+1])
			h.length += h.l[i]
		}
		if h.length == 0 {
			return FormatError("Huffman table has zero length")
		}
		if h.length > maxNumValues {
			return FormatError("Huffman table has excessive length")
		}
		n -= h.length + 17
		if n < 0 {
			return FormatError("DHT has wrong length")
		}
		_, err = io.ReadFull(d.r, h.val[0:h.length])
		if err != nil {
			return err
		}

		// Derive size.
		k := 0
		for i := 0; i < maxCodeLength; i++ {
			for j := 0; j < h.l[i]; j++ {
				h.size[k] = i + 1
				k++
			}
		}

		// Derive code.
		code := 0
		size := h.size[0]
		for i := 0; i < h.length; i++ {
			if size != h.size[i] {
				code <<= uint8(h.size[i] - size)
				size = h.size[i]
			}
			h.code[i] = code
			code++
		}

		// Derive minCode, maxCode, and valIndex.
		k = 0
		index := 0
		for i := 0; i < maxCodeLength; i++ {
			if h.l[i] == 0 {
				h.minCode[i] = -1
				h.maxCode[i] = -1
				h.valIndex[i] = -1
			} else {
				h.minCode[i] = k
				h.maxCode[i] = k + h.l[i] - 1
				h.valIndex[i] = index
				k += h.l[i]
				index += h.l[i]
			}
			k <<= 1
		}
	}
	return nil
}

// Returns the next Huffman-coded value from the bit stream, decoded according to h.
// TODO(nigeltao): This decoding algorithm is simple, but slow. A lookahead table, instead of always
// peeling off only 1 bit at at time, ought to be faster.
func (d *decoder) decodeHuffman(h *huffman) (uint8, os.Error) {
	if h.length == 0 {
		return 0, FormatError("uninitialized Huffman table")
	}
	for i, code := 0, 0; i < maxCodeLength; i++ {
		err := d.ensureNBits(1)
		if err != nil {
			return 0, err
		}
		if d.b.a&d.b.m != 0 {
			code |= 1
		}
		d.b.n--
		d.b.m >>= 1
		if code <= h.maxCode[i] {
			return h.val[h.valIndex[i]+code-h.minCode[i]], nil
		}
		code <<= 1
	}
	return 0, FormatError("bad Huffman code")
}
