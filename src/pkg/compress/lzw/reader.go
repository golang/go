// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package lzw implements the Lempel-Ziv-Welch compressed data format,
// described in T. A. Welch, ``A Technique for High-Performance Data
// Compression'', Computer, 17(6) (June 1984), pp 8-19.
//
// In particular, it implements LZW as used by the GIF, TIFF and PDF file
// formats, which means variable-width codes up to 12 bits and the first
// two non-literal codes are a clear code and an EOF code.
package lzw

// TODO(nigeltao): check that TIFF and PDF use LZW in the same way as GIF,
// modulo LSB/MSB packing order.

import (
	"bufio"
	"fmt"
	"io"
	"os"
)

// Order specifies the bit ordering in an LZW data stream.
type Order int

const (
	// LSB means Least Significant Bits first, as used in the GIF file format.
	LSB Order = iota
	// MSB means Most Significant Bits first, as used in the TIFF and PDF
	// file formats.
	MSB
)

// decoder is the state from which the readXxx method converts a byte
// stream into a code stream.
type decoder struct {
	r     io.ByteReader
	bits  uint32
	nBits uint
	width uint
}

// readLSB returns the next code for "Least Significant Bits first" data.
func (d *decoder) readLSB() (uint16, os.Error) {
	for d.nBits < d.width {
		x, err := d.r.ReadByte()
		if err != nil {
			return 0, err
		}
		d.bits |= uint32(x) << d.nBits
		d.nBits += 8
	}
	code := uint16(d.bits & (1<<d.width - 1))
	d.bits >>= d.width
	d.nBits -= d.width
	return code, nil
}

// readMSB returns the next code for "Most Significant Bits first" data.
func (d *decoder) readMSB() (uint16, os.Error) {
	for d.nBits < d.width {
		x, err := d.r.ReadByte()
		if err != nil {
			return 0, err
		}
		d.bits |= uint32(x) << (24 - d.nBits)
		d.nBits += 8
	}
	code := uint16(d.bits >> (32 - d.width))
	d.bits <<= d.width
	d.nBits -= d.width
	return code, nil
}

// decode decompresses bytes from r and writes them to pw.
// read specifies how to decode bytes into codes.
// litWidth is the width in bits of literal codes.
func decode(r io.Reader, read func(*decoder) (uint16, os.Error), litWidth int, pw *io.PipeWriter) {
	br, ok := r.(io.ByteReader)
	if !ok {
		br = bufio.NewReader(r)
	}
	pw.CloseWithError(decode1(pw, br, read, uint(litWidth)))
}

func decode1(pw *io.PipeWriter, r io.ByteReader, read func(*decoder) (uint16, os.Error), litWidth uint) os.Error {
	const (
		maxWidth    = 12
		invalidCode = 0xffff
	)
	d := decoder{r, 0, 0, 1 + litWidth}
	w := bufio.NewWriter(pw)
	// The first 1<<litWidth codes are literal codes.
	// The next two codes mean clear and EOF.
	// Other valid codes are in the range [lo, hi] where lo := clear + 2,
	// with the upper bound incrementing on each code seen.
	clear := uint16(1) << litWidth
	eof, hi := clear+1, clear+1
	// overflow is the code at which hi overflows the code width.
	overflow := uint16(1) << d.width
	var (
		// Each code c in [lo, hi] expands to two or more bytes. For c != hi:
		//   suffix[c] is the last of these bytes.
		//   prefix[c] is the code for all but the last byte.
		//   This code can either be a literal code or another code in [lo, c).
		// The c == hi case is a special case.
		suffix [1 << maxWidth]uint8
		prefix [1 << maxWidth]uint16
		// buf is a scratch buffer for reconstituting the bytes that a code expands to.
		// Code suffixes are written right-to-left from the end of the buffer.
		buf [1 << maxWidth]byte
	)

	// Loop over the code stream, converting codes into decompressed bytes.
	last := uint16(invalidCode)
	for {
		code, err := read(&d)
		if err != nil {
			if err == os.EOF {
				err = io.ErrUnexpectedEOF
			}
			return err
		}
		switch {
		case code < clear:
			// We have a literal code.
			if err := w.WriteByte(uint8(code)); err != nil {
				return err
			}
			if last != invalidCode {
				// Save what the hi code expands to.
				suffix[hi] = uint8(code)
				prefix[hi] = last
			}
		case code == clear:
			d.width = 1 + litWidth
			hi = eof
			overflow = 1 << d.width
			last = invalidCode
			continue
		case code == eof:
			return w.Flush()
		case code <= hi:
			c, i := code, len(buf)-1
			if code == hi {
				// code == hi is a special case which expands to the last expansion
				// followed by the head of the last expansion. To find the head, we walk
				// the prefix chain until we find a literal code.
				c = last
				for c >= clear {
					c = prefix[c]
				}
				buf[i] = uint8(c)
				i--
				c = last
			}
			// Copy the suffix chain into buf and then write that to w.
			for c >= clear {
				buf[i] = suffix[c]
				i--
				c = prefix[c]
			}
			buf[i] = uint8(c)
			if _, err := w.Write(buf[i:]); err != nil {
				return err
			}
			if last != invalidCode {
				// Save what the hi code expands to.
				suffix[hi] = uint8(c)
				prefix[hi] = last
			}
		default:
			return os.NewError("lzw: invalid code")
		}
		last, hi = code, hi+1
		if hi >= overflow {
			if d.width == maxWidth {
				last = invalidCode
				continue
			}
			d.width++
			overflow <<= 1
		}
	}
	panic("unreachable")
}

// NewReader creates a new io.ReadCloser that satisfies reads by decompressing
// the data read from r.
// It is the caller's responsibility to call Close on the ReadCloser when
// finished reading.
// The number of bits to use for literal codes, litWidth, must be in the
// range [2,8] and is typically 8.
func NewReader(r io.Reader, order Order, litWidth int) io.ReadCloser {
	pr, pw := io.Pipe()
	var read func(*decoder) (uint16, os.Error)
	switch order {
	case LSB:
		read = (*decoder).readLSB
	case MSB:
		read = (*decoder).readMSB
	default:
		pw.CloseWithError(os.NewError("lzw: unknown order"))
		return pr
	}
	if litWidth < 2 || 8 < litWidth {
		pw.CloseWithError(fmt.Errorf("lzw: litWidth %d out of range", litWidth))
		return pr
	}
	go decode(r, read, litWidth, pw)
	return pr
}
