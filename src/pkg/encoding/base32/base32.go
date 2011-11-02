// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package base32 implements base32 encoding as specified by RFC 4648.
package base32

import (
	"io"
	"strconv"
)

/*
 * Encodings
 */

// An Encoding is a radix 32 encoding/decoding scheme, defined by a
// 32-character alphabet.  The most common is the "base32" encoding
// introduced for SASL GSSAPI and standardized in RFC 4648.
// The alternate "base32hex" encoding is used in DNSSEC.
type Encoding struct {
	encode    string
	decodeMap [256]byte
}

const encodeStd = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"
const encodeHex = "0123456789ABCDEFGHIJKLMNOPQRSTUV"

// NewEncoding returns a new Encoding defined by the given alphabet,
// which must be a 32-byte string.
func NewEncoding(encoder string) *Encoding {
	e := new(Encoding)
	e.encode = encoder
	for i := 0; i < len(e.decodeMap); i++ {
		e.decodeMap[i] = 0xFF
	}
	for i := 0; i < len(encoder); i++ {
		e.decodeMap[encoder[i]] = byte(i)
	}
	return e
}

// StdEncoding is the standard base32 encoding, as defined in
// RFC 4648.
var StdEncoding = NewEncoding(encodeStd)

// HexEncoding is the ``Extended Hex Alphabet'' defined in RFC 4648.
// It is typically used in DNS.
var HexEncoding = NewEncoding(encodeHex)

/*
 * Encoder
 */

// Encode encodes src using the encoding enc, writing
// EncodedLen(len(src)) bytes to dst.
//
// The encoding pads the output to a multiple of 8 bytes,
// so Encode is not appropriate for use on individual blocks
// of a large data stream.  Use NewEncoder() instead.
func (enc *Encoding) Encode(dst, src []byte) {
	if len(src) == 0 {
		return
	}

	for len(src) > 0 {
		dst[0] = 0
		dst[1] = 0
		dst[2] = 0
		dst[3] = 0
		dst[4] = 0
		dst[5] = 0
		dst[6] = 0
		dst[7] = 0

		// Unpack 8x 5-bit source blocks into a 5 byte
		// destination quantum
		switch len(src) {
		default:
			dst[7] |= src[4] & 0x1F
			dst[6] |= src[4] >> 5
			fallthrough
		case 4:
			dst[6] |= (src[3] << 3) & 0x1F
			dst[5] |= (src[3] >> 2) & 0x1F
			dst[4] |= src[3] >> 7
			fallthrough
		case 3:
			dst[4] |= (src[2] << 1) & 0x1F
			dst[3] |= (src[2] >> 4) & 0x1F
			fallthrough
		case 2:
			dst[3] |= (src[1] << 4) & 0x1F
			dst[2] |= (src[1] >> 1) & 0x1F
			dst[1] |= (src[1] >> 6) & 0x1F
			fallthrough
		case 1:
			dst[1] |= (src[0] << 2) & 0x1F
			dst[0] |= src[0] >> 3
		}

		// Encode 5-bit blocks using the base32 alphabet
		for j := 0; j < 8; j++ {
			dst[j] = enc.encode[dst[j]]
		}

		// Pad the final quantum
		if len(src) < 5 {
			dst[7] = '='
			if len(src) < 4 {
				dst[6] = '='
				dst[5] = '='
				if len(src) < 3 {
					dst[4] = '='
					if len(src) < 2 {
						dst[3] = '='
						dst[2] = '='
					}
				}
			}
			break
		}
		src = src[5:]
		dst = dst[8:]
	}
}

type encoder struct {
	err  error
	enc  *Encoding
	w    io.Writer
	buf  [5]byte    // buffered data waiting to be encoded
	nbuf int        // number of bytes in buf
	out  [1024]byte // output buffer
}

func (e *encoder) Write(p []byte) (n int, err error) {
	if e.err != nil {
		return 0, e.err
	}

	// Leading fringe.
	if e.nbuf > 0 {
		var i int
		for i = 0; i < len(p) && e.nbuf < 5; i++ {
			e.buf[e.nbuf] = p[i]
			e.nbuf++
		}
		n += i
		p = p[i:]
		if e.nbuf < 5 {
			return
		}
		e.enc.Encode(e.out[0:], e.buf[0:])
		if _, e.err = e.w.Write(e.out[0:8]); e.err != nil {
			return n, e.err
		}
		e.nbuf = 0
	}

	// Large interior chunks.
	for len(p) >= 5 {
		nn := len(e.out) / 8 * 5
		if nn > len(p) {
			nn = len(p)
		}
		nn -= nn % 5
		if nn > 0 {
			e.enc.Encode(e.out[0:], p[0:nn])
			if _, e.err = e.w.Write(e.out[0 : nn/5*8]); e.err != nil {
				return n, e.err
			}
		}
		n += nn
		p = p[nn:]
	}

	// Trailing fringe.
	for i := 0; i < len(p); i++ {
		e.buf[i] = p[i]
	}
	e.nbuf = len(p)
	n += len(p)
	return
}

// Close flushes any pending output from the encoder.
// It is an error to call Write after calling Close.
func (e *encoder) Close() error {
	// If there's anything left in the buffer, flush it out
	if e.err == nil && e.nbuf > 0 {
		e.enc.Encode(e.out[0:], e.buf[0:e.nbuf])
		e.nbuf = 0
		_, e.err = e.w.Write(e.out[0:8])
	}
	return e.err
}

// NewEncoder returns a new base32 stream encoder.  Data written to
// the returned writer will be encoded using enc and then written to w.
// Base32 encodings operate in 5-byte blocks; when finished
// writing, the caller must Close the returned encoder to flush any
// partially written blocks.
func NewEncoder(enc *Encoding, w io.Writer) io.WriteCloser {
	return &encoder{enc: enc, w: w}
}

// EncodedLen returns the length in bytes of the base32 encoding
// of an input buffer of length n.
func (enc *Encoding) EncodedLen(n int) int { return (n + 4) / 5 * 8 }

/*
 * Decoder
 */

type CorruptInputError int64

func (e CorruptInputError) Error() string {
	return "illegal base32 data at input byte " + strconv.Itoa64(int64(e))
}

// decode is like Decode but returns an additional 'end' value, which
// indicates if end-of-message padding was encountered and thus any
// additional data is an error.  decode also assumes len(src)%8==0,
// since it is meant for internal use.
func (enc *Encoding) decode(dst, src []byte) (n int, end bool, err error) {
	for i := 0; i < len(src)/8 && !end; i++ {
		// Decode quantum using the base32 alphabet
		var dbuf [8]byte
		dlen := 8

		// do the top bytes contain any data?
	dbufloop:
		for j := 0; j < 8; j++ {
			in := src[i*8+j]
			if in == '=' && j >= 2 && i == len(src)/8-1 {
				// We've reached the end and there's
				// padding, the rest should be padded
				for k := j; k < 8; k++ {
					if src[i*8+k] != '=' {
						return n, false, CorruptInputError(i*8 + j)
					}
				}
				dlen = j
				end = true
				break dbufloop
			}
			dbuf[j] = enc.decodeMap[in]
			if dbuf[j] == 0xFF {
				return n, false, CorruptInputError(i*8 + j)
			}
		}

		// Pack 8x 5-bit source blocks into 5 byte destination
		// quantum
		switch dlen {
		case 7, 8:
			dst[i*5+4] = dbuf[6]<<5 | dbuf[7]
			fallthrough
		case 6, 5:
			dst[i*5+3] = dbuf[4]<<7 | dbuf[5]<<2 | dbuf[6]>>3
			fallthrough
		case 4:
			dst[i*5+2] = dbuf[3]<<4 | dbuf[4]>>1
			fallthrough
		case 3:
			dst[i*5+1] = dbuf[1]<<6 | dbuf[2]<<1 | dbuf[3]>>4
			fallthrough
		case 2:
			dst[i*5+0] = dbuf[0]<<3 | dbuf[1]>>2
		}
		switch dlen {
		case 2:
			n += 1
		case 3, 4:
			n += 2
		case 5:
			n += 3
		case 6, 7:
			n += 4
		case 8:
			n += 5
		}
	}
	return n, end, nil
}

// Decode decodes src using the encoding enc.  It writes at most
// DecodedLen(len(src)) bytes to dst and returns the number of bytes
// written.  If src contains invalid base32 data, it will return the
// number of bytes successfully written and CorruptInputError.
func (enc *Encoding) Decode(dst, src []byte) (n int, err error) {
	if len(src)%8 != 0 {
		return 0, CorruptInputError(len(src) / 8 * 8)
	}

	n, _, err = enc.decode(dst, src)
	return
}

type decoder struct {
	err    error
	enc    *Encoding
	r      io.Reader
	end    bool       // saw end of message
	buf    [1024]byte // leftover input
	nbuf   int
	out    []byte // leftover decoded output
	outbuf [1024 / 8 * 5]byte
}

func (d *decoder) Read(p []byte) (n int, err error) {
	if d.err != nil {
		return 0, d.err
	}

	// Use leftover decoded output from last read.
	if len(d.out) > 0 {
		n = copy(p, d.out)
		d.out = d.out[n:]
		return n, nil
	}

	// Read a chunk.
	nn := len(p) / 5 * 8
	if nn < 8 {
		nn = 8
	}
	if nn > len(d.buf) {
		nn = len(d.buf)
	}
	nn, d.err = io.ReadAtLeast(d.r, d.buf[d.nbuf:nn], 8-d.nbuf)
	d.nbuf += nn
	if d.nbuf < 8 {
		return 0, d.err
	}

	// Decode chunk into p, or d.out and then p if p is too small.
	nr := d.nbuf / 8 * 8
	nw := d.nbuf / 8 * 5
	if nw > len(p) {
		nw, d.end, d.err = d.enc.decode(d.outbuf[0:], d.buf[0:nr])
		d.out = d.outbuf[0:nw]
		n = copy(p, d.out)
		d.out = d.out[n:]
	} else {
		n, d.end, d.err = d.enc.decode(p, d.buf[0:nr])
	}
	d.nbuf -= nr
	for i := 0; i < d.nbuf; i++ {
		d.buf[i] = d.buf[i+nr]
	}

	if d.err == nil {
		d.err = err
	}
	return n, d.err
}

// NewDecoder constructs a new base32 stream decoder.
func NewDecoder(enc *Encoding, r io.Reader) io.Reader {
	return &decoder{enc: enc, r: r}
}

// DecodedLen returns the maximum length in bytes of the decoded data
// corresponding to n bytes of base32-encoded data.
func (enc *Encoding) DecodedLen(n int) int { return n / 8 * 5 }
