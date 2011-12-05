// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package git85 implements the radix 85 data encoding
// used in the Git version control system.
package git85

import (
	"bytes"
	"io"
	"strconv"
)

type CorruptInputError int64

func (e CorruptInputError) Error() string {
	return "illegal git85 data at input byte " + strconv.FormatInt(int64(e), 10)
}

const encode = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!#$%&()*+-;<=>?@^_`{|}~"

// The decodings are 1+ the actual value, so that the
// default zero value can be used to mean "not valid".
var decode = [256]uint8{
	'0': 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
	'A': 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
	24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
	'a': 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
	50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
	'!': 63,
	'#': 64, 65, 66, 67,
	'(': 68, 69, 70, 71,
	'-': 72,
	';': 73,
	'<': 74, 75, 76, 77,
	'@': 78,
	'^': 79, 80, 81,
	'{': 82, 83, 84, 85,
}

// Encode encodes src into EncodedLen(len(src))
// bytes of dst.  As a convenience, it returns the number
// of bytes written to dst, but this value is always EncodedLen(len(src)).
// Encode implements the radix 85 encoding used in the
// Git version control tool.
//
// The encoding splits src into chunks of at most 52 bytes
// and encodes each chunk on its own line.
func Encode(dst, src []byte) int {
	ndst := 0
	for len(src) > 0 {
		n := len(src)
		if n > 52 {
			n = 52
		}
		if n <= 27 {
			dst[ndst] = byte('A' + n - 1)
		} else {
			dst[ndst] = byte('a' + n - 26 - 1)
		}
		ndst++
		for i := 0; i < n; i += 4 {
			var v uint32
			for j := 0; j < 4 && i+j < n; j++ {
				v |= uint32(src[i+j]) << uint(24-j*8)
			}
			for j := 4; j >= 0; j-- {
				dst[ndst+j] = encode[v%85]
				v /= 85
			}
			ndst += 5
		}
		dst[ndst] = '\n'
		ndst++
		src = src[n:]
	}
	return ndst
}

// EncodedLen returns the length of an encoding of n source bytes.
func EncodedLen(n int) int {
	if n == 0 {
		return 0
	}
	// 5 bytes per 4 bytes of input, rounded up.
	// 2 extra bytes for each line of 52 src bytes, rounded up.
	return (n+3)/4*5 + (n+51)/52*2
}

var newline = []byte{'\n'}

// Decode decodes src into at most MaxDecodedLen(len(src))
// bytes, returning the actual number of bytes written to dst.
//
// If Decode encounters invalid input, it returns a CorruptInputError.
//
func Decode(dst, src []byte) (n int, err error) {
	ndst := 0
	nsrc := 0
	for nsrc < len(src) {
		var l int
		switch ch := int(src[nsrc]); {
		case 'A' <= ch && ch <= 'Z':
			l = ch - 'A' + 1
		case 'a' <= ch && ch <= 'z':
			l = ch - 'a' + 26 + 1
		default:
			return ndst, CorruptInputError(nsrc)
		}
		if nsrc+1+l > len(src) {
			return ndst, CorruptInputError(nsrc)
		}
		el := (l + 3) / 4 * 5 // encoded len
		if nsrc+1+el+1 > len(src) || src[nsrc+1+el] != '\n' {
			return ndst, CorruptInputError(nsrc)
		}
		line := src[nsrc+1 : nsrc+1+el]
		for i := 0; i < el; i += 5 {
			var v uint32
			for j := 0; j < 5; j++ {
				ch := decode[line[i+j]]
				if ch == 0 {
					return ndst, CorruptInputError(nsrc + 1 + i + j)
				}
				v = v*85 + uint32(ch-1)
			}
			for j := 0; j < 4; j++ {
				dst[ndst] = byte(v >> 24)
				v <<= 8
				ndst++
			}
		}
		// Last fragment may have run too far (but there was room in dst).
		// Back up.
		if l%4 != 0 {
			ndst -= 4 - l%4
		}
		nsrc += 1 + el + 1
	}
	return ndst, nil
}

func MaxDecodedLen(n int) int { return n / 5 * 4 }

// NewEncoder returns a new Git base85 stream encoder.  Data written to
// the returned writer will be encoded and then written to w.
// The Git encoding operates on 52-byte blocks; when finished
// writing, the caller must Close the returned encoder to flush any
// partially written blocks.
func NewEncoder(w io.Writer) io.WriteCloser { return &encoder{w: w} }

type encoder struct {
	w    io.Writer
	err  error
	buf  [52]byte
	nbuf int
	out  [1024]byte
	nout int
}

func (e *encoder) Write(p []byte) (n int, err error) {
	if e.err != nil {
		return 0, e.err
	}

	// Leading fringe.
	if e.nbuf > 0 {
		var i int
		for i = 0; i < len(p) && e.nbuf < 52; i++ {
			e.buf[e.nbuf] = p[i]
			e.nbuf++
		}
		n += i
		p = p[i:]
		if e.nbuf < 52 {
			return
		}
		nout := Encode(e.out[0:], e.buf[0:])
		if _, e.err = e.w.Write(e.out[0:nout]); e.err != nil {
			return n, e.err
		}
		e.nbuf = 0
	}

	// Large interior chunks.
	for len(p) >= 52 {
		nn := len(e.out) / (1 + 52/4*5 + 1) * 52
		if nn > len(p) {
			nn = len(p) / 52 * 52
		}
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
	for i := 0; i < len(p); i++ {
		e.buf[i] = p[i]
	}
	e.nbuf = len(p)
	n += len(p)
	return
}

func (e *encoder) Close() error {
	// If there's anything left in the buffer, flush it out
	if e.err == nil && e.nbuf > 0 {
		nout := Encode(e.out[0:], e.buf[0:e.nbuf])
		e.nbuf = 0
		_, e.err = e.w.Write(e.out[0:nout])
	}
	return e.err
}

// NewDecoder returns a new Git base85 stream decoder.
func NewDecoder(r io.Reader) io.Reader { return &decoder{r: r} }

type decoder struct {
	r       io.Reader
	err     error
	readErr error
	buf     [1024]byte
	nbuf    int
	out     []byte
	outbuf  [1024]byte
	off     int64
}

func (d *decoder) Read(p []byte) (n int, err error) {
	if len(p) == 0 {
		return 0, nil
	}

	for {
		// Copy leftover output from last decode.
		if len(d.out) > 0 {
			n = copy(p, d.out)
			d.out = d.out[n:]
			return
		}

		// Out of decoded output.  Check errors.
		if d.err != nil {
			return 0, d.err
		}
		if d.readErr != nil {
			d.err = d.readErr
			return 0, d.err
		}

		// Read and decode more input.
		var nn int
		nn, d.readErr = d.r.Read(d.buf[d.nbuf:])
		d.nbuf += nn

		// Send complete lines to Decode.
		nl := bytes.LastIndex(d.buf[0:d.nbuf], newline)
		if nl < 0 {
			continue
		}
		nn, d.err = Decode(d.outbuf[0:], d.buf[0:nl+1])
		if e, ok := d.err.(CorruptInputError); ok {
			d.err = CorruptInputError(int64(e) + d.off)
		}
		d.out = d.outbuf[0:nn]
		d.nbuf = copy(d.buf[0:], d.buf[nl+1:d.nbuf])
		d.off += int64(nl + 1)
	}
	panic("unreachable")
}
