// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package ascii85 implements the ascii85 data encoding
// as used in the btoa tool and Adobe's PostScript and PDF document formats.
package ascii85

import (
	"io"
	"strconv"
)

/*
 * Encoder
 */

// Encode encodes src into at most MaxEncodedLen(len(src))
// bytes of dst, returning the actual number of bytes written.
//
// The encoding handles 4-byte chunks, using a special encoding
// for the last fragment, so Encode is not appropriate for use on
// individual blocks of a large data stream. Use NewEncoder() instead.
//
// Often, ascii85-encoded data is wrapped in <~ and ~> symbols.
// Encode does not add these.
func Encode(dst, src []byte) int {
	if len(src) == 0 {
		return 0
	}

	n := 0
	for len(src) > 0 {
		dst[0] = 0
		dst[1] = 0
		dst[2] = 0
		dst[3] = 0
		dst[4] = 0

		// Unpack 4 bytes into uint32 to repack into base 85 5-byte.
		var v uint32
		switch len(src) {
		default:
			v |= uint32(src[3])
			fallthrough
		case 3:
			v |= uint32(src[2]) << 8
			fallthrough
		case 2:
			v |= uint32(src[1]) << 16
			fallthrough
		case 1:
			v |= uint32(src[0]) << 24
		}

		// Special case: zero (!!!!!) shortens to z.
		if v == 0 && len(src) >= 4 {
			dst[0] = 'z'
			dst = dst[1:]
			src = src[4:]
			n++
			continue
		}

		// Otherwise, 5 base 85 digits starting at !.
		for i := 4; i >= 0; i-- {
			dst[i] = '!' + byte(v%85)
			v /= 85
		}

		// If src was short, discard the low destination bytes.
		m := 5
		if len(src) < 4 {
			m -= 4 - len(src)
			src = nil
		} else {
			src = src[4:]
		}
		dst = dst[m:]
		n += m
	}
	return n
}

// MaxEncodedLen returns the maximum length of an encoding of n source bytes.
func MaxEncodedLen(n int) int { return (n + 3) / 4 * 5 }

// NewEncoder returns a new ascii85 stream encoder. Data written to
// the returned writer will be encoded and then written to w.
// Ascii85 encodings operate in 32-bit blocks; when finished
// writing, the caller must Close the returned encoder to flush any
// trailing partial block.
func NewEncoder(w io.Writer) io.WriteCloser { return &encoder{w: w} }

type encoder struct {
	err  error
	w    io.Writer
	buf  [4]byte    // buffered data waiting to be encoded
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
		for i = 0; i < len(p) && e.nbuf < 4; i++ {
			e.buf[e.nbuf] = p[i]
			e.nbuf++
		}
		n += i
		p = p[i:]
		if e.nbuf < 4 {
			return
		}
		nout := Encode(e.out[0:], e.buf[0:])
		if _, e.err = e.w.Write(e.out[0:nout]); e.err != nil {
			return n, e.err
		}
		e.nbuf = 0
	}

	// Large interior chunks.
	for len(p) >= 4 {
		nn := len(e.out) / 5 * 4
		if nn > len(p) {
			nn = len(p)
		}
		nn -= nn % 4
		if nn > 0 {
			nout := Encode(e.out[0:], p[0:nn])
			if _, e.err = e.w.Write(e.out[0:nout]); e.err != nil {
				return n, e.err
			}
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
		nout := Encode(e.out[0:], e.buf[0:e.nbuf])
		e.nbuf = 0
		_, e.err = e.w.Write(e.out[0:nout])
	}
	return e.err
}

/*
 * Decoder
 */

type CorruptInputError int64

func (e CorruptInputError) Error() string {
	return "illegal ascii85 data at input byte " + strconv.FormatInt(int64(e), 10)
}

// Decode decodes src into dst, returning both the number
// of bytes written to dst and the number consumed from src.
// If src contains invalid ascii85 data, Decode will return the
// number of bytes successfully written and a CorruptInputError.
// Decode ignores space and control characters in src.
// Often, ascii85-encoded data is wrapped in <~ and ~> symbols.
// Decode expects these to have been stripped by the caller.
//
// If flush is true, Decode assumes that src represents the
// end of the input stream and processes it completely rather
// than wait for the completion of another 32-bit block.
//
// NewDecoder wraps an io.Reader interface around Decode.
func Decode(dst, src []byte, flush bool) (ndst, nsrc int, err error) {
	var v uint32
	var nb int
	for i, b := range src {
		if len(dst)-ndst < 4 {
			return
		}
		switch {
		case b <= ' ':
			continue
		case b == 'z' && nb == 0:
			nb = 5
			v = 0
		case '!' <= b && b <= 'u':
			v = v*85 + uint32(b-'!')
			nb++
		default:
			return 0, 0, CorruptInputError(i)
		}
		if nb == 5 {
			nsrc = i + 1
			dst[ndst] = byte(v >> 24)
			dst[ndst+1] = byte(v >> 16)
			dst[ndst+2] = byte(v >> 8)
			dst[ndst+3] = byte(v)
			ndst += 4
			nb = 0
			v = 0
		}
	}
	if flush {
		nsrc = len(src)
		if nb > 0 {
			// The number of output bytes in the last fragment
			// is the number of leftover input bytes - 1:
			// the extra byte provides enough bits to cover
			// the inefficiency of the encoding for the block.
			if nb == 1 {
				return 0, 0, CorruptInputError(len(src))
			}
			for i := nb; i < 5; i++ {
				// The short encoding truncated the output value.
				// We have to assume the worst case values (digit 84)
				// in order to ensure that the top bits are correct.
				v = v*85 + 84
			}
			for i := 0; i < nb-1; i++ {
				dst[ndst] = byte(v >> 24)
				v <<= 8
				ndst++
			}
		}
	}
	return
}

// NewDecoder constructs a new ascii85 stream decoder.
func NewDecoder(r io.Reader) io.Reader { return &decoder{r: r} }

type decoder struct {
	err     error
	readErr error
	r       io.Reader
	buf     [1024]byte // leftover input
	nbuf    int
	out     []byte // leftover decoded output
	outbuf  [1024]byte
}

func (d *decoder) Read(p []byte) (n int, err error) {
	if len(p) == 0 {
		return 0, nil
	}
	if d.err != nil {
		return 0, d.err
	}

	for {
		// Copy leftover output from last decode.
		if len(d.out) > 0 {
			n = copy(p, d.out)
			d.out = d.out[n:]
			return
		}

		// Decode leftover input from last read.
		var nn, nsrc, ndst int
		if d.nbuf > 0 {
			ndst, nsrc, d.err = Decode(d.outbuf[0:], d.buf[0:d.nbuf], d.readErr != nil)
			if ndst > 0 {
				d.out = d.outbuf[0:ndst]
				d.nbuf = copy(d.buf[0:], d.buf[nsrc:d.nbuf])
				continue // copy out and return
			}
			if ndst == 0 && d.err == nil {
				// Special case: input buffer is mostly filled with non-data bytes.
				// Filter out such bytes to make room for more input.
				off := 0
				for i := 0; i < d.nbuf; i++ {
					if d.buf[i] > ' ' {
						d.buf[off] = d.buf[i]
						off++
					}
				}
				d.nbuf = off
			}
		}

		// Out of input, out of decoded output. Check errors.
		if d.err != nil {
			return 0, d.err
		}
		if d.readErr != nil {
			d.err = d.readErr
			return 0, d.err
		}

		// Read more data.
		nn, d.readErr = d.r.Read(d.buf[d.nbuf:])
		d.nbuf += nn
	}
}
