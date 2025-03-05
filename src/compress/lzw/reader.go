// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package lzw implements the Lempel-Ziv-Welch compressed data format,
// described in T. A. Welch, “A Technique for High-Performance Data
// Compression”, Computer, 17(6) (June 1984), pp 8-19.
//
// In particular, it implements LZW as used by the GIF and PDF file
// formats, which means variable-width codes up to 12 bits and the first
// two non-literal codes are a clear code and an EOF code.
//
// The TIFF file format uses a similar but incompatible version of the LZW
// algorithm. See the [golang.org/x/image/tiff/lzw] package for an
// implementation.
package lzw

// TODO(nigeltao): check that PDF uses LZW in the same way as GIF,
// modulo LSB/MSB packing order.

import (
	"bufio"
	"errors"
	"fmt"
	"io"
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

const (
	maxWidth           = 12
	decoderInvalidCode = 0xffff
	flushBuffer        = 1 << maxWidth
)

// Reader is an [io.Reader] which can be used to read compressed data in the
// LZW format.
type Reader struct {
	r        io.ByteReader
	bits     uint32
	nBits    uint
	width    uint
	read     func(*Reader) (uint16, error) // readLSB or readMSB
	litWidth int                           // width in bits of literal codes
	err      error

	// The first 1<<litWidth codes are literal codes.
	// The next two codes mean clear and EOF.
	// Other valid codes are in the range [lo, hi] where lo := clear + 2,
	// with the upper bound incrementing on each code seen.
	//
	// overflow is the code at which hi overflows the code width. It always
	// equals 1 << width.
	//
	// last is the most recently seen code, or decoderInvalidCode.
	//
	// An invariant is that hi < overflow.
	clear, eof, hi, overflow, last uint16

	// Each code c in [lo, hi] expands to two or more bytes. For c != hi:
	//   suffix[c] is the last of these bytes.
	//   prefix[c] is the code for all but the last byte.
	//   This code can either be a literal code or another code in [lo, c).
	// The c == hi case is a special case.
	suffix [1 << maxWidth]uint8
	prefix [1 << maxWidth]uint16

	// output is the temporary output buffer.
	// Literal codes are accumulated from the start of the buffer.
	// Non-literal codes decode to a sequence of suffixes that are first
	// written right-to-left from the end of the buffer before being copied
	// to the start of the buffer.
	// It is flushed when it contains >= 1<<maxWidth bytes,
	// so that there is always room to decode an entire code.
	output [2 * 1 << maxWidth]byte
	o      int    // write index into output
	toRead []byte // bytes to return from Read
}

// readLSB returns the next code for "Least Significant Bits first" data.
func (r *Reader) readLSB() (uint16, error) {
	for r.nBits < r.width {
		x, err := r.r.ReadByte()
		if err != nil {
			return 0, err
		}
		r.bits |= uint32(x) << r.nBits
		r.nBits += 8
	}
	code := uint16(r.bits & (1<<r.width - 1))
	r.bits >>= r.width
	r.nBits -= r.width
	return code, nil
}

// readMSB returns the next code for "Most Significant Bits first" data.
func (r *Reader) readMSB() (uint16, error) {
	for r.nBits < r.width {
		x, err := r.r.ReadByte()
		if err != nil {
			return 0, err
		}
		r.bits |= uint32(x) << (24 - r.nBits)
		r.nBits += 8
	}
	code := uint16(r.bits >> (32 - r.width))
	r.bits <<= r.width
	r.nBits -= r.width
	return code, nil
}

// Read implements io.Reader, reading uncompressed bytes from its underlying [Reader].
func (r *Reader) Read(b []byte) (int, error) {
	for {
		if len(r.toRead) > 0 {
			n := copy(b, r.toRead)
			r.toRead = r.toRead[n:]
			return n, nil
		}
		if r.err != nil {
			return 0, r.err
		}
		r.decode()
	}
}

// decode decompresses bytes from r and leaves them in d.toRead.
// read specifies how to decode bytes into codes.
// litWidth is the width in bits of literal codes.
func (r *Reader) decode() {
	// Loop over the code stream, converting codes into decompressed bytes.
loop:
	for {
		code, err := r.read(r)
		if err != nil {
			if err == io.EOF {
				err = io.ErrUnexpectedEOF
			}
			r.err = err
			break
		}
		switch {
		case code < r.clear:
			// We have a literal code.
			r.output[r.o] = uint8(code)
			r.o++
			if r.last != decoderInvalidCode {
				// Save what the hi code expands to.
				r.suffix[r.hi] = uint8(code)
				r.prefix[r.hi] = r.last
			}
		case code == r.clear:
			r.width = 1 + uint(r.litWidth)
			r.hi = r.eof
			r.overflow = 1 << r.width
			r.last = decoderInvalidCode
			continue
		case code == r.eof:
			r.err = io.EOF
			break loop
		case code <= r.hi:
			c, i := code, len(r.output)-1
			if code == r.hi && r.last != decoderInvalidCode {
				// code == hi is a special case which expands to the last expansion
				// followed by the head of the last expansion. To find the head, we walk
				// the prefix chain until we find a literal code.
				c = r.last
				for c >= r.clear {
					c = r.prefix[c]
				}
				r.output[i] = uint8(c)
				i--
				c = r.last
			}
			// Copy the suffix chain into output and then write that to w.
			for c >= r.clear {
				r.output[i] = r.suffix[c]
				i--
				c = r.prefix[c]
			}
			r.output[i] = uint8(c)
			r.o += copy(r.output[r.o:], r.output[i:])
			if r.last != decoderInvalidCode {
				// Save what the hi code expands to.
				r.suffix[r.hi] = uint8(c)
				r.prefix[r.hi] = r.last
			}
		default:
			r.err = errors.New("lzw: invalid code")
			break loop
		}
		r.last, r.hi = code, r.hi+1
		if r.hi >= r.overflow {
			if r.hi > r.overflow {
				panic("unreachable")
			}
			if r.width == maxWidth {
				r.last = decoderInvalidCode
				// Undo the d.hi++ a few lines above, so that (1) we maintain
				// the invariant that d.hi < d.overflow, and (2) d.hi does not
				// eventually overflow a uint16.
				r.hi--
			} else {
				r.width++
				r.overflow = 1 << r.width
			}
		}
		if r.o >= flushBuffer {
			break
		}
	}
	// Flush pending output.
	r.toRead = r.output[:r.o]
	r.o = 0
}

var errClosed = errors.New("lzw: reader/writer is closed")

// Close closes the [Reader] and returns an error for any future read operation.
// It does not close the underlying [io.Reader].
func (r *Reader) Close() error {
	r.err = errClosed // in case any Reads come along
	return nil
}

// Reset clears the [Reader]'s state and allows it to be reused again
// as a new [Reader].
func (r *Reader) Reset(src io.Reader, order Order, litWidth int) {
	*r = Reader{}
	r.init(src, order, litWidth)
}

// NewReader creates a new [io.ReadCloser].
// Reads from the returned [io.ReadCloser] read and decompress data from r.
// If r does not also implement [io.ByteReader],
// the decompressor may read more data than necessary from r.
// It is the caller's responsibility to call Close on the ReadCloser when
// finished reading.
// The number of bits to use for literal codes, litWidth, must be in the
// range [2,8] and is typically 8. It must equal the litWidth
// used during compression.
//
// It is guaranteed that the underlying type of the returned [io.ReadCloser]
// is a *[Reader].
func NewReader(r io.Reader, order Order, litWidth int) io.ReadCloser {
	return newReader(r, order, litWidth)
}

func newReader(src io.Reader, order Order, litWidth int) *Reader {
	r := new(Reader)
	r.init(src, order, litWidth)
	return r
}

func (r *Reader) init(src io.Reader, order Order, litWidth int) {
	switch order {
	case LSB:
		r.read = (*Reader).readLSB
	case MSB:
		r.read = (*Reader).readMSB
	default:
		r.err = errors.New("lzw: unknown order")
		return
	}
	if litWidth < 2 || 8 < litWidth {
		r.err = fmt.Errorf("lzw: litWidth %d out of range", litWidth)
		return
	}

	br, ok := src.(io.ByteReader)
	if !ok && src != nil {
		br = bufio.NewReader(src)
	}
	r.r = br
	r.litWidth = litWidth
	r.width = 1 + uint(litWidth)
	r.clear = uint16(1) << uint(litWidth)
	r.eof, r.hi = r.clear+1, r.clear+1
	r.overflow = uint16(1) << r.width
	r.last = decoderInvalidCode
}
