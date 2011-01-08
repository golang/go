// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This package implements a Reader which handles reading \r and \r\n
// deliminated lines.
package line

import (
	"io"
	"os"
)

// Reader reads lines from an io.Reader (which may use either '\n' or
// '\r\n').
type Reader struct {
	buf      []byte
	consumed int
	in       io.Reader
	err      os.Error
}

func NewReader(in io.Reader, maxLineLength int) *Reader {
	return &Reader{
		buf:      make([]byte, 0, maxLineLength),
		consumed: 0,
		in:       in,
	}
}

// ReadLine tries to return a single line, not including the end-of-line bytes.
// If the line was found to be longer than the maximum length then isPrefix is
// set and the beginning of the line is returned. The rest of the line will be
// returned from future calls. isPrefix will be false when returning the last
// fragment of the line.  The returned buffer points into the internal state of
// the Reader and is only valid until the next call to ReadLine. ReadLine
// either returns a non-nil line or it returns an error, never both.
func (l *Reader) ReadLine() (line []byte, isPrefix bool, err os.Error) {
	if l.consumed > 0 {
		n := copy(l.buf, l.buf[l.consumed:])
		l.buf = l.buf[:n]
		l.consumed = 0
	}

	if len(l.buf) == 0 && l.err != nil {
		err = l.err
		return
	}

	scannedTo := 0

	for {
		i := scannedTo
		for ; i < len(l.buf); i++ {
			if l.buf[i] == '\r' && len(l.buf) > i+1 && l.buf[i+1] == '\n' {
				line = l.buf[:i]
				l.consumed = i + 2
				return
			} else if l.buf[i] == '\n' {
				line = l.buf[:i]
				l.consumed = i + 1
				return
			}
		}

		if i == cap(l.buf) {
			line = l.buf[:i]
			l.consumed = i
			isPrefix = true
			return
		}

		if l.err != nil {
			line = l.buf
			l.consumed = i
			return
		}

		// We don't want to rescan the input that we just scanned.
		// However, we need to back up one byte because the last byte
		// could have been a '\r' and we do need to rescan that.
		scannedTo = i
		if scannedTo > 0 {
			scannedTo--
		}
		oldLen := len(l.buf)
		l.buf = l.buf[:cap(l.buf)]
		n, readErr := l.in.Read(l.buf[oldLen:])
		l.buf = l.buf[:oldLen+n]
		if readErr != nil {
			l.err = readErr
		}
	}
	panic("unreachable")
}
