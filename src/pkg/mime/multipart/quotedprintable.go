// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The file define a quoted-printable decoder, as specified in RFC 2045.

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

func (q *qpReader) Read(p []byte) (n int, err error) {
	for len(p) > 0 {
		if len(q.line) == 0 {
			if q.rerr != nil {
				return n, q.rerr
			}
			q.line, q.rerr = q.br.ReadSlice('\n')
			q.line = bytes.TrimRightFunc(q.line, isQPDiscardWhitespace)
			continue
		}
		if len(q.line) == 1 && q.line[0] == '=' {
			// Soft newline; skipped.
			q.line = nil
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
		case b != '\t' && (b < ' ' || b > '~'):
			return n, fmt.Errorf("multipart: invalid unescaped byte 0x%02x in quoted-printable body", b)
		}
		p[0] = b
		p = p[1:]
		q.line = q.line[1:]
		n++
	}
	return n, nil
}
