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
// 32-character alphabet. The most common is the "base32" encoding
// introduced for SASL GSSAPI and standardized in RFC 4648.
// The alternate "base32hex" encoding is used in DNSSEC.
type Encoding struct {
	encode    [32]byte
	decodeMap [256]byte
	padChar   rune
}

const (
	StdPadding rune = '=' // Standard padding character
	NoPadding  rune = -1  // No padding
)

const encodeStd = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"
const encodeHex = "0123456789ABCDEFGHIJKLMNOPQRSTUV"

// NewEncoding returns a new Encoding defined by the given alphabet,
// which must be a 32-byte string.
func NewEncoding(encoder string) *Encoding {
	if len(encoder) != 32 {
		panic("encoding alphabet is not 32-bytes long")
	}

	e := new(Encoding)
	copy(e.encode[:], encoder)
	e.padChar = StdPadding

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

// WithPadding creates a new encoding identical to enc except
// with a specified padding character, or NoPadding to disable padding.
// The padding character must not be '\r' or '\n', must not
// be contained in the encoding's alphabet and must be a rune equal or
// below '\xff'.
func (enc Encoding) WithPadding(padding rune) *Encoding {
	if padding == '\r' || padding == '\n' || padding > 0xff {
		panic("invalid padding")
	}

	for i := 0; i < len(enc.encode); i++ {
		if rune(enc.encode[i]) == padding {
			panic("padding contained in alphabet")
		}
	}

	enc.padChar = padding
	return &enc
}

/*
 * Encoder
 */

// Encode encodes src using the encoding enc, writing
// EncodedLen(len(src)) bytes to dst.
//
// The encoding pads the output to a multiple of 8 bytes,
// so Encode is not appropriate for use on individual blocks
// of a large data stream. Use NewEncoder() instead.
func (enc *Encoding) Encode(dst, src []byte) {
	for len(src) > 0 {
		var b [8]byte

		// Unpack 8x 5-bit source blocks into a 5 byte
		// destination quantum
		switch len(src) {
		default:
			b[7] = src[4] & 0x1F
			b[6] = src[4] >> 5
			fallthrough
		case 4:
			b[6] |= (src[3] << 3) & 0x1F
			b[5] = (src[3] >> 2) & 0x1F
			b[4] = src[3] >> 7
			fallthrough
		case 3:
			b[4] |= (src[2] << 1) & 0x1F
			b[3] = (src[2] >> 4) & 0x1F
			fallthrough
		case 2:
			b[3] |= (src[1] << 4) & 0x1F
			b[2] = (src[1] >> 1) & 0x1F
			b[1] = (src[1] >> 6) & 0x1F
			fallthrough
		case 1:
			b[1] |= (src[0] << 2) & 0x1F
			b[0] = src[0] >> 3
		}

		// Encode 5-bit blocks using the base32 alphabet
		size := len(dst)
		if size >= 8 {
			// Common case, unrolled for extra performance
			dst[0] = enc.encode[b[0]&31]
			dst[1] = enc.encode[b[1]&31]
			dst[2] = enc.encode[b[2]&31]
			dst[3] = enc.encode[b[3]&31]
			dst[4] = enc.encode[b[4]&31]
			dst[5] = enc.encode[b[5]&31]
			dst[6] = enc.encode[b[6]&31]
			dst[7] = enc.encode[b[7]&31]
		} else {
			for i := 0; i < size; i++ {
				dst[i] = enc.encode[b[i]&31]
			}
		}

		// Pad the final quantum
		if len(src) < 5 {
			if enc.padChar == NoPadding {
				break
			}

			dst[7] = byte(enc.padChar)
			if len(src) < 4 {
				dst[6] = byte(enc.padChar)
				dst[5] = byte(enc.padChar)
				if len(src) < 3 {
					dst[4] = byte(enc.padChar)
					if len(src) < 2 {
						dst[3] = byte(enc.padChar)
						dst[2] = byte(enc.padChar)
					}
				}
			}

			break
		}

		src = src[5:]
		dst = dst[8:]
	}
}

// EncodeToString returns the base32 encoding of src.
func (enc *Encoding) EncodeToString(src []byte) string {
	buf := make([]byte, enc.EncodedLen(len(src)))
	enc.Encode(buf, src)
	return string(buf)
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
			nn -= nn % 5
		}
		e.enc.Encode(e.out[0:], p[0:nn])
		if _, e.err = e.w.Write(e.out[0 : nn/5*8]); e.err != nil {
			return n, e.err
		}
		n += nn
		p = p[nn:]
	}

	// Trailing fringe.
	copy(e.buf[:], p)
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
		encodedLen := e.enc.EncodedLen(e.nbuf)
		e.nbuf = 0
		_, e.err = e.w.Write(e.out[0:encodedLen])
	}
	return e.err
}

// NewEncoder returns a new base32 stream encoder. Data written to
// the returned writer will be encoded using enc and then written to w.
// Base32 encodings operate in 5-byte blocks; when finished
// writing, the caller must Close the returned encoder to flush any
// partially written blocks.
func NewEncoder(enc *Encoding, w io.Writer) io.WriteCloser {
	return &encoder{enc: enc, w: w}
}

// EncodedLen returns the length in bytes of the base32 encoding
// of an input buffer of length n.
func (enc *Encoding) EncodedLen(n int) int {
	if enc.padChar == NoPadding {
		return (n*8 + 4) / 5
	}
	return (n + 4) / 5 * 8
}

/*
 * Decoder
 */

type CorruptInputError int64

func (e CorruptInputError) Error() string {
	return "illegal base32 data at input byte " + strconv.FormatInt(int64(e), 10)
}

// decode is like Decode but returns an additional 'end' value, which
// indicates if end-of-message padding was encountered and thus any
// additional data is an error. This method assumes that src has been
// stripped of all supported whitespace ('\r' and '\n').
func (enc *Encoding) decode(dst, src []byte) (n int, end bool, err error) {
	// Lift the nil check outside of the loop.
	_ = enc.decodeMap

	dsti := 0
	olen := len(src)

	for len(src) > 0 && !end {
		// Decode quantum using the base32 alphabet
		var dbuf [8]byte
		dlen := 8

		for j := 0; j < 8; {

			if len(src) == 0 {
				if enc.padChar != NoPadding {
					// We have reached the end and are missing padding
					return n, false, CorruptInputError(olen - len(src) - j)
				}
				// We have reached the end and are not expecting any padding
				dlen, end = j, true
				break
			}
			in := src[0]
			src = src[1:]
			if in == byte(enc.padChar) && j >= 2 && len(src) < 8 {
				// We've reached the end and there's padding
				if len(src)+j < 8-1 {
					// not enough padding
					return n, false, CorruptInputError(olen)
				}
				for k := 0; k < 8-1-j; k++ {
					if len(src) > k && src[k] != byte(enc.padChar) {
						// incorrect padding
						return n, false, CorruptInputError(olen - len(src) + k - 1)
					}
				}
				dlen, end = j, true
				// 7, 5 and 2 are not valid padding lengths, and so 1, 3 and 6 are not
				// valid dlen values. See RFC 4648 Section 6 "Base 32 Encoding" listing
				// the five valid padding lengths, and Section 9 "Illustrations and
				// Examples" for an illustration for how the 1st, 3rd and 6th base32
				// src bytes do not yield enough information to decode a dst byte.
				if dlen == 1 || dlen == 3 || dlen == 6 {
					return n, false, CorruptInputError(olen - len(src) - 1)
				}
				break
			}
			dbuf[j] = enc.decodeMap[in]
			if dbuf[j] == 0xFF {
				return n, false, CorruptInputError(olen - len(src) - 1)
			}
			j++
		}

		// Pack 8x 5-bit source blocks into 5 byte destination
		// quantum
		switch dlen {
		case 8:
			dst[dsti+4] = dbuf[6]<<5 | dbuf[7]
			n++
			fallthrough
		case 7:
			dst[dsti+3] = dbuf[4]<<7 | dbuf[5]<<2 | dbuf[6]>>3
			n++
			fallthrough
		case 5:
			dst[dsti+2] = dbuf[3]<<4 | dbuf[4]>>1
			n++
			fallthrough
		case 4:
			dst[dsti+1] = dbuf[1]<<6 | dbuf[2]<<1 | dbuf[3]>>4
			n++
			fallthrough
		case 2:
			dst[dsti+0] = dbuf[0]<<3 | dbuf[1]>>2
			n++
		}
		dsti += 5
	}
	return n, end, nil
}

// Decode decodes src using the encoding enc. It writes at most
// DecodedLen(len(src)) bytes to dst and returns the number of bytes
// written. If src contains invalid base32 data, it will return the
// number of bytes successfully written and CorruptInputError.
// New line characters (\r and \n) are ignored.
func (enc *Encoding) Decode(dst, src []byte) (n int, err error) {
	buf := make([]byte, len(src))
	l := stripNewlines(buf, src)
	n, _, err = enc.decode(dst, buf[:l])
	return
}

// DecodeString returns the bytes represented by the base32 string s.
func (enc *Encoding) DecodeString(s string) ([]byte, error) {
	buf := []byte(s)
	l := stripNewlines(buf, buf)
	n, _, err := enc.decode(buf, buf[:l])
	return buf[:n], err
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

func readEncodedData(r io.Reader, buf []byte, min int, expectsPadding bool) (n int, err error) {
	for n < min && err == nil {
		var nn int
		nn, err = r.Read(buf[n:])
		n += nn
	}
	// data was read, less than min bytes could be read
	if n < min && n > 0 && err == io.EOF {
		err = io.ErrUnexpectedEOF
	}
	// no data was read, the buffer already contains some data
	// when padding is disabled this is not an error, as the message can be of
	// any length
	if expectsPadding && min < 8 && n == 0 && err == io.EOF {
		err = io.ErrUnexpectedEOF
	}
	return
}

func (d *decoder) Read(p []byte) (n int, err error) {
	// Use leftover decoded output from last read.
	if len(d.out) > 0 {
		n = copy(p, d.out)
		d.out = d.out[n:]
		if len(d.out) == 0 {
			return n, d.err
		}
		return n, nil
	}

	if d.err != nil {
		return 0, d.err
	}

	// Read a chunk.
	nn := len(p) / 5 * 8
	if nn < 8 {
		nn = 8
	}
	if nn > len(d.buf) {
		nn = len(d.buf)
	}

	// Minimum amount of bytes that needs to be read each cycle
	var min int
	var expectsPadding bool
	if d.enc.padChar == NoPadding {
		min = 1
		expectsPadding = false
	} else {
		min = 8 - d.nbuf
		expectsPadding = true
	}

	nn, d.err = readEncodedData(d.r, d.buf[d.nbuf:nn], min, expectsPadding)
	d.nbuf += nn
	if d.nbuf < min {
		return 0, d.err
	}

	// Decode chunk into p, or d.out and then p if p is too small.
	var nr int
	if d.enc.padChar == NoPadding {
		nr = d.nbuf
	} else {
		nr = d.nbuf / 8 * 8
	}
	nw := d.enc.DecodedLen(d.nbuf)

	if nw > len(p) {
		nw, d.end, err = d.enc.decode(d.outbuf[0:], d.buf[0:nr])
		d.out = d.outbuf[0:nw]
		n = copy(p, d.out)
		d.out = d.out[n:]
	} else {
		n, d.end, err = d.enc.decode(p, d.buf[0:nr])
	}
	d.nbuf -= nr
	for i := 0; i < d.nbuf; i++ {
		d.buf[i] = d.buf[i+nr]
	}

	if err != nil && (d.err == nil || d.err == io.EOF) {
		d.err = err
	}

	if len(d.out) > 0 {
		// We cannot return all the decoded bytes to the caller in this
		// invocation of Read, so we return a nil error to ensure that Read
		// will be called again.  The error stored in d.err, if any, will be
		// returned with the last set of decoded bytes.
		return n, nil
	}

	return n, d.err
}

type newlineFilteringReader struct {
	wrapped io.Reader
}

// stripNewlines removes newline characters and returns the number
// of non-newline characters copied to dst.
func stripNewlines(dst, src []byte) int {
	offset := 0
	for _, b := range src {
		if b == '\r' || b == '\n' {
			continue
		}
		dst[offset] = b
		offset++
	}
	return offset
}

func (r *newlineFilteringReader) Read(p []byte) (int, error) {
	n, err := r.wrapped.Read(p)
	for n > 0 {
		s := p[0:n]
		offset := stripNewlines(s, s)
		if err != nil || offset > 0 {
			return offset, err
		}
		// Previous buffer entirely whitespace, read again
		n, err = r.wrapped.Read(p)
	}
	return n, err
}

// NewDecoder constructs a new base32 stream decoder.
func NewDecoder(enc *Encoding, r io.Reader) io.Reader {
	return &decoder{enc: enc, r: &newlineFilteringReader{r}}
}

// DecodedLen returns the maximum length in bytes of the decoded data
// corresponding to n bytes of base32-encoded data.
func (enc *Encoding) DecodedLen(n int) int {
	if enc.padChar == NoPadding {
		return n * 5 / 8
	}

	return n / 8 * 5
}
