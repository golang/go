// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package quotedprintable implements quoted-printable encoding as specified by
// RFC 2045.
package quotedprintable

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
)

// Reader is a quoted-printable decoder.
type Reader struct {
	br   *bufio.Reader
	rerr error  // last read error
	line []byte // to be consumed before more of br
}

// NewReader returns a quoted-printable reader, decoding from r.
func NewReader(r io.Reader) *Reader {
	return &Reader{
		br: bufio.NewReader(r),
	}
}

func fromHex(b byte) (byte, error) {
	switch {
	case b >= '0' && b <= '9':
		return b - '0', nil
	case b >= 'A' && b <= 'F':
		return b - 'A' + 10, nil
	// Accept badly encoded bytes.
	case b >= 'a' && b <= 'f':
		return b - 'a' + 10, nil
	}
	return 0, fmt.Errorf("quotedprintable: invalid hex byte 0x%02x", b)
}

func readHexByte(v []byte) (b byte, err error) {
	if len(v) < 2 {
		return 0, io.ErrUnexpectedEOF
	}
	var hb, lb byte
	if hb, err = fromHex(v[0]); err != nil {
		return 0, err
	}
	if lb, err = fromHex(v[1]); err != nil {
		return 0, err
	}
	return hb<<4 | lb, nil
}

func isQPDiscardWhitespace(r rune) bool {
	switch r {
	case '\n', '\r', ' ', '\t':
		return true
	}
	return false
}

var (
	crlf       = []byte("\r\n")
	lf         = []byte("\n")
	softSuffix = []byte("=")
)

// Read reads and decodes quoted-printable data from the underlying reader.
func (r *Reader) Read(p []byte) (n int, err error) {
	// Deviations from RFC 2045:
	// 1. in addition to "=\r\n", "=\n" is also treated as soft line break.
	// 2. it will pass through a '\r' or '\n' not preceded by '=', consistent
	//    with other broken QP encoders & decoders.
	// 3. it accepts soft line-break (=) at end of message (issue 15486); i.e.
	//    the final byte read from the underlying reader is allowed to be '=',
	//    and it will be silently ignored.
	// 4. it takes = as literal = if not followed by two hex digits
	//    but not at end of line (issue 13219).
	for len(p) > 0 {
		if len(r.line) == 0 {
			if r.rerr != nil {
				return n, r.rerr
			}
			r.line, r.rerr = r.br.ReadSlice('\n')

			// Does the line end in CRLF instead of just LF?
			hasLF := bytes.HasSuffix(r.line, lf)
			hasCR := bytes.HasSuffix(r.line, crlf)
			wholeLine := r.line
			r.line = bytes.TrimRightFunc(wholeLine, isQPDiscardWhitespace)
			if bytes.HasSuffix(r.line, softSuffix) {
				rightStripped := wholeLine[len(r.line):]
				r.line = r.line[:len(r.line)-1]
				if !bytes.HasPrefix(rightStripped, lf) && !bytes.HasPrefix(rightStripped, crlf) &&
					!(len(rightStripped) == 0 && len(r.line) > 0 && r.rerr == io.EOF) {
					r.rerr = fmt.Errorf("quotedprintable: invalid bytes after =: %q", rightStripped)
				}
			} else if hasLF {
				if hasCR {
					r.line = append(r.line, '\r', '\n')
				} else {
					r.line = append(r.line, '\n')
				}
			}
			continue
		}
		b := r.line[0]

		switch {
		case b == '=':
			b, err = readHexByte(r.line[1:])
			if err != nil {
				if len(r.line) >= 2 && r.line[1] != '\r' && r.line[1] != '\n' {
					// Take the = as a literal =.
					b = '='
					break
				}
				return n, err
			}
			r.line = r.line[2:] // 2 of the 3; other 1 is done below
		case b == '\t' || b == '\r' || b == '\n':
			break
		case b >= 0x80:
			// As an extension to RFC 2045, we accept
			// values >= 0x80 without complaint. Issue 22597.
			break
		case b < ' ' || b > '~':
			return n, fmt.Errorf("quotedprintable: invalid unescaped byte 0x%02x in body", b)
		}
		p[0] = b
		p = p[1:]
		r.line = r.line[1:]
		n++
	}
	return n, nil
}
