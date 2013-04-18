// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The file define a quoted-printable decoder, as specified in RFC 2045.
// Deviations:
// 1. in addition to "=\r\n", "=\n" is also treated as soft line break.
// 2. it will pass through a '\r' or '\n' not preceded by '=', consistent
//    with other broken QP encoders & decoders.

package multipart

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
)

type qpReader struct {
	br   *bufio.Reader
	rerr error  // last read error
	line []byte // to be consumed before more of br
}

func newQuotedPrintableReader(r io.Reader) io.Reader {
	return &qpReader{
		br: bufio.NewReader(r),
	}
}

func fromHex(b byte) (byte, error) {
	switch {
	case b >= '0' && b <= '9':
		return b - '0', nil
	case b >= 'A' && b <= 'F':
		return b - 'A' + 10, nil
	}
	return 0, fmt.Errorf("multipart: invalid quoted-printable hex byte 0x%02x", b)
}

func (q *qpReader) readHexByte(v []byte) (b byte, err error) {
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

func (q *qpReader) Read(p []byte) (n int, err error) {
	for len(p) > 0 {
		if len(q.line) == 0 {
			if q.rerr != nil {
				return n, q.rerr
			}
			q.line, q.rerr = q.br.ReadSlice('\n')

			// Does the line end in CRLF instead of just LF?
			hasLF := bytes.HasSuffix(q.line, lf)
			hasCR := bytes.HasSuffix(q.line, crlf)
			wholeLine := q.line
			q.line = bytes.TrimRightFunc(wholeLine, isQPDiscardWhitespace)
			if bytes.HasSuffix(q.line, softSuffix) {
				rightStripped := wholeLine[len(q.line):]
				q.line = q.line[:len(q.line)-1]
				if !bytes.HasPrefix(rightStripped, lf) && !bytes.HasPrefix(rightStripped, crlf) {
					q.rerr = fmt.Errorf("multipart: invalid bytes after =: %q", rightStripped)
				}
			} else if hasLF {
				if hasCR {
					q.line = append(q.line, '\r', '\n')
				} else {
					q.line = append(q.line, '\n')
				}
			}
			continue
		}
		b := q.line[0]

		switch {
		case b == '=':
			b, err = q.readHexByte(q.line[1:])
			if err != nil {
				return n, err
			}
			q.line = q.line[2:] // 2 of the 3; other 1 is done below
		case b == '\t' || b == '\r' || b == '\n':
			break
		case b < ' ' || b > '~':
			return n, fmt.Errorf("multipart: invalid unescaped byte 0x%02x in quoted-printable body", b)
		}
		p[0] = b
		p = p[1:]
		q.line = q.line[1:]
		n++
	}
	return n, nil
}
