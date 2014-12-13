// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package base64 implements base64 encoding as specified by RFC 4648.
package base64

import (
	"bytes"
	"io"
	"strconv"
	"strings"
)

/*
 * Encodings
 */

// An Encoding is a radix 64 encoding/decoding scheme, defined by a
// 64-character alphabet.  The most common encoding is the "base64"
// encoding defined in RFC 4648 and used in MIME (RFC 2045) and PEM
// (RFC 1421).  RFC 4648 also defines an alternate encoding, which is
// the standard encoding with - and _ substituted for + and /.
type Encoding struct {
	encode    string
	decodeMap [256]byte
	padChar   rune
}

const (
	StdPadding rune = '=' // Standard padding character
	NoPadding  rune = -1  // No padding
)

const encodeStd = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
const encodeURL = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"

// NewEncoding returns a new padded Encoding defined by the given alphabet,
// which must be a 64-byte string.
// The resulting Encoding uses the default padding character ('='),
// which may be changed or disabled via WithPadding.
func NewEncoding(encoder string) *Encoding {
	e := new(Encoding)
	e.encode = encoder
	e.padChar = StdPadding
	for i := 0; i < len(e.decodeMap); i++ {
		e.decodeMap[i] = 0xFF
	}
	for i := 0; i < len(encoder); i++ {
		e.decodeMap[encoder[i]] = byte(i)
	}
	return e
}

// WithPadding creates a new encoding identical to enc except
// with a specified padding character, or NoPadding to disable padding.
func (enc Encoding) WithPadding(padding rune) *Encoding {
	enc.padChar = padding
	return &enc
}

// StdEncoding is the standard base64 encoding, as defined in
// RFC 4648.
var StdEncoding = NewEncoding(encodeStd)

// URLEncoding is the alternate base64 encoding defined in RFC 4648.
// It is typically used in URLs and file names.
var URLEncoding = NewEncoding(encodeURL)

// RawStdEncoding is the standard raw, unpadded base64 encoding,
// as defined in RFC 4648 section 3.2.
// This is the same as StdEncoding but omits padding characters.
var RawStdEncoding = StdEncoding.WithPadding(NoPadding)

// URLEncoding is the unpadded alternate base64 encoding defined in RFC 4648.
// It is typically used in URLs and file names.
// This is the same as URLEncoding but omits padding characters.
var RawURLEncoding = URLEncoding.WithPadding(NoPadding)

var removeNewlinesMapper = func(r rune) rune {
	if r == '\r' || r == '\n' {
		return -1
	}
	return r
}

/*
 * Encoder
 */

// Encode encodes src using the encoding enc, writing
// EncodedLen(len(src)) bytes to dst.
//
// The encoding pads the output to a multiple of 4 bytes,
// so Encode is not appropriate for use on individual blocks
// of a large data stream.  Use NewEncoder() instead.
func (enc *Encoding) Encode(dst, src []byte) {
	if len(src) == 0 {
		return
	}

	for len(src) > 0 {
		var b0, b1, b2, b3 byte

		// Unpack 4x 6-bit source blocks into a 4 byte
		// destination quantum
		switch len(src) {
		default:
			b3 = src[2] & 0x3F
			b2 = src[2] >> 6
			fallthrough
		case 2:
			b2 |= (src[1] << 2) & 0x3F
			b1 = src[1] >> 4
			fallthrough
		case 1:
			b1 |= (src[0] << 4) & 0x3F
			b0 = src[0] >> 2
		}

		// Encode 6-bit blocks using the base64 alphabet
		dst[0] = enc.encode[b0]
		dst[1] = enc.encode[b1]
		if len(src) >= 3 {
			dst[2] = enc.encode[b2]
			dst[3] = enc.encode[b3]
		} else { // Final incomplete quantum
			if len(src) >= 2 {
				dst[2] = enc.encode[b2]
			}
			if enc.padChar != NoPadding {
				if len(src) < 2 {
					dst[2] = byte(enc.padChar)
				}
				dst[3] = byte(enc.padChar)
			}
			break
		}

		src = src[3:]
		dst = dst[4:]
	}
}

// EncodeToString returns the base64 encoding of src.
func (enc *Encoding) EncodeToString(src []byte) string {
	buf := make([]byte, enc.EncodedLen(len(src)))
	enc.Encode(buf, src)
	return string(buf)
}

type encoder struct {
	err  error
	enc  *Encoding
	w    io.Writer
	buf  [3]byte    // buffered data waiting to be encoded
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
		for i = 0; i < len(p) && e.nbuf < 3; i++ {
			e.buf[e.nbuf] = p[i]
			e.nbuf++
		}
		n += i
		p = p[i:]
		if e.nbuf < 3 {
			return
		}
		e.enc.Encode(e.out[:], e.buf[:])
		if _, e.err = e.w.Write(e.out[:4]); e.err != nil {
			return n, e.err
		}
		e.nbuf = 0
	}

	// Large interior chunks.
	for len(p) >= 3 {
		nn := len(e.out) / 4 * 3
		if nn > len(p) {
			nn = len(p)
			nn -= nn % 3
		}
		e.enc.Encode(e.out[:], p[:nn])
		if _, e.err = e.w.Write(e.out[0 : nn/3*4]); e.err != nil {
			return n, e.err
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
		e.enc.Encode(e.out[:], e.buf[:e.nbuf])
		_, e.err = e.w.Write(e.out[:e.enc.EncodedLen(e.nbuf)])
		e.nbuf = 0
	}
	return e.err
}

// NewEncoder returns a new base64 stream encoder.  Data written to
// the returned writer will be encoded using enc and then written to w.
// Base64 encodings operate in 4-byte blocks; when finished
// writing, the caller must Close the returned encoder to flush any
// partially written blocks.
func NewEncoder(enc *Encoding, w io.Writer) io.WriteCloser {
	return &encoder{enc: enc, w: w}
}

// EncodedLen returns the length in bytes of the base64 encoding
// of an input buffer of length n.
func (enc *Encoding) EncodedLen(n int) int {
	if enc.padChar == NoPadding {
		return (n*8 + 5) / 6 // minimum # chars at 6 bits per char
	}
	return (n + 2) / 3 * 4 // minimum # 4-char quanta, 3 bytes each
}

/*
 * Decoder
 */

type CorruptInputError int64

func (e CorruptInputError) Error() string {
	return "illegal base64 data at input byte " + strconv.FormatInt(int64(e), 10)
}

// decode is like Decode but returns an additional 'end' value, which
// indicates if end-of-message padding or a partial quantum was encountered
// and thus any additional data is an error. This method assumes that src has been
// stripped of all supported whitespace ('\r' and '\n').
func (enc *Encoding) decode(dst, src []byte) (n int, end bool, err error) {
	olen := len(src)
	for len(src) > 0 && !end {
		// Decode quantum using the base64 alphabet
		var dbuf [4]byte
		dinc, dlen := 3, 4

		for j := range dbuf {
			if len(src) == 0 {
				if enc.padChar != NoPadding || j < 2 {
					return n, false, CorruptInputError(olen - len(src) - j)
				}
				dinc, dlen, end = j-1, j, true
				break
			}
			in := src[0]
			src = src[1:]
			if rune(in) == enc.padChar {
				// We've reached the end and there's padding
				switch j {
				case 0, 1:
					// incorrect padding
					return n, false, CorruptInputError(olen - len(src) - 1)
				case 2:
					// "==" is expected, the first "=" is already consumed.
					if len(src) == 0 {
						// not enough padding
						return n, false, CorruptInputError(olen)
					}
					if rune(src[0]) != enc.padChar {
						// incorrect padding
						return n, false, CorruptInputError(olen - len(src) - 1)
					}
					src = src[1:]
				}
				if len(src) > 0 {
					// trailing garbage
					err = CorruptInputError(olen - len(src))
				}
				dinc, dlen, end = 3, j, true
				break
			}
			dbuf[j] = enc.decodeMap[in]
			if dbuf[j] == 0xFF {
				return n, false, CorruptInputError(olen - len(src) - 1)
			}
		}

		// Pack 4x 6-bit source blocks into 3 byte destination
		// quantum
		switch dlen {
		case 4:
			dst[2] = dbuf[2]<<6 | dbuf[3]
			fallthrough
		case 3:
			dst[1] = dbuf[1]<<4 | dbuf[2]>>2
			fallthrough
		case 2:
			dst[0] = dbuf[0]<<2 | dbuf[1]>>4
		}
		dst = dst[dinc:]
		n += dlen - 1
	}

	return n, end, err
}

// Decode decodes src using the encoding enc.  It writes at most
// DecodedLen(len(src)) bytes to dst and returns the number of bytes
// written.  If src contains invalid base64 data, it will return the
// number of bytes successfully written and CorruptInputError.
// New line characters (\r and \n) are ignored.
func (enc *Encoding) Decode(dst, src []byte) (n int, err error) {
	src = bytes.Map(removeNewlinesMapper, src)
	n, _, err = enc.decode(dst, src)
	return
}

// DecodeString returns the bytes represented by the base64 string s.
func (enc *Encoding) DecodeString(s string) ([]byte, error) {
	s = strings.Map(removeNewlinesMapper, s)
	dbuf := make([]byte, enc.DecodedLen(len(s)))
	n, _, err := enc.decode(dbuf, []byte(s))
	return dbuf[:n], err
}

type decoder struct {
	err    error
	enc    *Encoding
	r      io.Reader
	end    bool       // saw end of message
	buf    [1024]byte // leftover input
	nbuf   int
	out    []byte // leftover decoded output
	outbuf [1024 / 4 * 3]byte
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
	nn := len(p) / 3 * 4
	if nn < 4 {
		nn = 4
	}
	if nn > len(d.buf) {
		nn = len(d.buf)
	}
	nn, d.err = io.ReadAtLeast(d.r, d.buf[d.nbuf:nn], 4-d.nbuf)
	d.nbuf += nn
	if d.err != nil || d.nbuf < 4 {
		return 0, d.err
	}

	// Decode chunk into p, or d.out and then p if p is too small.
	nr := d.nbuf / 4 * 4
	nw := d.nbuf / 4 * 3
	if nw > len(p) {
		nw, d.end, d.err = d.enc.decode(d.outbuf[:], d.buf[:nr])
		d.out = d.outbuf[:nw]
		n = copy(p, d.out)
		d.out = d.out[n:]
	} else {
		n, d.end, d.err = d.enc.decode(p, d.buf[:nr])
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

type newlineFilteringReader struct {
	wrapped io.Reader
}

func (r *newlineFilteringReader) Read(p []byte) (int, error) {
	n, err := r.wrapped.Read(p)
	for n > 0 {
		offset := 0
		for i, b := range p[:n] {
			if b != '\r' && b != '\n' {
				if i != offset {
					p[offset] = b
				}
				offset++
			}
		}
		if offset > 0 {
			return offset, err
		}
		// Previous buffer entirely whitespace, read again
		n, err = r.wrapped.Read(p)
	}
	return n, err
}

// NewDecoder constructs a new base64 stream decoder.
func NewDecoder(enc *Encoding, r io.Reader) io.Reader {
	return &decoder{enc: enc, r: &newlineFilteringReader{r}}
}

// DecodedLen returns the maximum length in bytes of the decoded data
// corresponding to n bytes of base64-encoded data.
func (enc *Encoding) DecodedLen(n int) int {
	if enc.padChar == NoPadding {
		// Unpadded data may end with partial block of 2-3 characters.
		return (n*6 + 7) / 8
	}
	// Padded base64 should always be a multiple of 4 characters in length.
	return n / 4 * 3
}
