// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package quotedprintable

import "io"

const lineMaxLen = 76

// A Writer is a quoted-printable writer that implements [io.WriteCloser].
type Writer struct {
	// Binary mode treats the writer's input as pure binary and processes end of
	// line bytes as binary data.
	Binary bool

	w    io.Writer
	i    int
	line [78]byte
	cr   bool
}

// NewWriter returns a new [Writer] that writes to w.
func NewWriter(w io.Writer) *Writer {
	return &Writer{w: w}
}

// Write encodes p using quoted-printable encoding and writes it to the
// underlying [io.Writer]. It limits line length to 76 characters. The encoded
// bytes are not necessarily flushed until the [Writer] is closed.
func (w *Writer) Write(p []byte) (n int, err error) {
	for i, b := range p {
		switch {
		// Simple writes are done in batch.
		case b >= '!' && b <= '~' && b != '=':
			continue
		case isWhitespace(b) || !w.Binary && (b == '\n' || b == '\r'):
			continue
		}

		if i > n {
			if err := w.write(p[n:i]); err != nil {
				return n, err
			}
			n = i
		}

		if err := w.encode(b); err != nil {
			return n, err
		}
		n++
	}

	if n == len(p) {
		return n, nil
	}

	if err := w.write(p[n:]); err != nil {
		return n, err
	}

	return len(p), nil
}

// Close closes the [Writer], flushing any unwritten data to the underlying
// [io.Writer], but does not close the underlying io.Writer.
func (w *Writer) Close() error {
	if err := w.checkLastByte(); err != nil {
		return err
	}

	return w.flush()
}

// write limits text encoded in quoted-printable to 76 characters per line.
func (w *Writer) write(p []byte) error {
	for _, b := range p {
		if b == '\n' || b == '\r' {
			// If the previous byte was \r, the CRLF has already been inserted.
			if w.cr && b == '\n' {
				w.cr = false
				continue
			}

			if b == '\r' {
				w.cr = true
			}

			if err := w.checkLastByte(); err != nil {
				return err
			}
			if err := w.insertCRLF(); err != nil {
				return err
			}
			continue
		}

		if w.i == lineMaxLen-1 {
			if err := w.insertSoftLineBreak(); err != nil {
				return err
			}
		}

		w.line[w.i] = b
		w.i++
		w.cr = false
	}

	return nil
}

func (w *Writer) encode(b byte) error {
	if lineMaxLen-1-w.i < 3 {
		if err := w.insertSoftLineBreak(); err != nil {
			return err
		}
	}

	w.line[w.i] = '='
	w.line[w.i+1] = upperhex[b>>4]
	w.line[w.i+2] = upperhex[b&0x0f]
	w.i += 3

	return nil
}

const upperhex = "0123456789ABCDEF"

// checkLastByte encodes the last buffered byte if it is a space or a tab.
func (w *Writer) checkLastByte() error {
	if w.i == 0 {
		return nil
	}

	b := w.line[w.i-1]
	if isWhitespace(b) {
		w.i--
		if err := w.encode(b); err != nil {
			return err
		}
	}

	return nil
}

func (w *Writer) insertSoftLineBreak() error {
	w.line[w.i] = '='
	w.i++

	return w.insertCRLF()
}

func (w *Writer) insertCRLF() error {
	w.line[w.i] = '\r'
	w.line[w.i+1] = '\n'
	w.i += 2

	return w.flush()
}

func (w *Writer) flush() error {
	if _, err := w.w.Write(w.line[:w.i]); err != nil {
		return err
	}

	w.i = 0
	return nil
}

func isWhitespace(b byte) bool {
	return b == ' ' || b == '\t'
}
