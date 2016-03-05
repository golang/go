// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import (
	"io"
	"unicode/utf8"
)

type source struct {
	src io.Reader
	end int

	buf             []byte
	litbuf          []byte
	pos, line       int
	oldpos, oldline int
	pin             int
}

func (s *source) init(src []byte) {
	s.buf = append(src, utf8.RuneSelf) // terminate with sentinel
	s.pos = 0
	s.line = 1
	s.oldline = 1
}

func (s *source) ungetr() {
	s.pos, s.line = s.oldpos, s.oldline
}

func (s *source) getr() rune {
redo:
	s.oldpos, s.oldline = s.pos, s.line

	// common case: 7bit ASCII
	if b := s.buf[s.pos]; b < utf8.RuneSelf {
		s.pos++
		if b == 0 {
			panic("invalid NUL byte")
			goto redo // (or return 0?)
		}
		if b == '\n' {
			s.line++
		}
		return rune(b)
	}

	// uncommon case: not ASCII or not enough bytes
	r, w := utf8.DecodeRune(s.buf[s.pos:])
	s.pos += w
	if r == utf8.RuneError && w == 1 {
		if s.pos >= len(s.buf) {
			s.ungetr() // so next getr also returns EOF
			return -1  // EOF
		}
		panic("invalid Unicode character")
		goto redo
	}

	// BOM's are only allowed as the first character in a file
	const BOM = 0xfeff
	if r == BOM && s.oldpos > 0 {
		panic("invalid BOM in the middle of the file")
		goto redo
	}

	return r
}

// TODO(gri) enable this one
func (s *source) getr_() rune {
redo:
	s.oldpos, s.oldline = s.pos, s.line

	// common case: 7bit ASCII
	if b := s.buf[s.pos]; b < utf8.RuneSelf {
		s.pos++
		if b == 0 {
			panic("invalid NUL byte")
			goto redo // (or return 0?)
		}
		if b == '\n' {
			s.line++
		}
		return rune(b)
	}

	// uncommon case: not ASCII or not enough bytes
	r, w := utf8.DecodeRune(s.buf[s.pos:s.end])
	if r == utf8.RuneError && w == 1 {
		if s.refill() {
			goto redo
		}
		// TODO(gri) carefull: this depends on whether s.end includes sentinel or not
		if s.pos < s.end {
			panic("invalid Unicode character")
			goto redo
		}
		// EOF
		return -1
	}

	s.pos += w

	// BOM's are only allowed as the first character in a file
	const BOM = 0xfeff
	if r == BOM && s.oldpos > 0 {
		panic("invalid BOM in the middle of the file")
		goto redo
	}

	return r
}

func (s *source) refill() bool {
	for s.pos+utf8.UTFMax > s.end && !utf8.FullRune(s.buf[s.pos:s.end]) {
		// not enough bytes

		// save literal prefix if any
		if s.pin >= 0 {
			s.litbuf = append(s.litbuf, s.buf[s.pin:s.oldpos]...)
			s.pin = 0
		}

		// move unread bytes to beginning of buffer
		copy(s.buf[0:], s.buf[s.oldpos:s.end])
		// read more bytes
		// (an io.Reader must return io.EOF when it reaches
		// the end of what it is reading - simply returning
		// n == 0 will make this loop retry forever; but the
		// error is in the reader implementation in that case)
		// TODO(gri) check for it and return io.ErrNoProgress?
		// (see also bufio.go:666)
		i := s.end - s.oldpos
		n, err := s.src.Read(s.buf[i : len(s.buf)-1])
		s.pos -= s.oldpos
		s.oldpos = 0
		s.end = i + n
		s.buf[s.end] = utf8.RuneSelf // sentinel
		if err != nil {
			if s.pos == s.end {
				return false // EOF
			}
			if err != io.EOF {
				panic(err) // TODO(gri) fix this
			}
			// If err == EOF, we won't be getting more
			// bytes; break to avoid infinite loop. If
			// err is something else, we don't know if
			// we can get more bytes; thus also break.
			break
		}
	}
	return true
}

func (s *source) startLit() {
	s.litbuf = s.litbuf[:0]
	s.pin = s.oldpos
}

func (s *source) stopLit() string {
	return string(s.buf[s.pin:s.pos])

	lit := s.buf[s.pin:s.pos]
	s.pin = -1
	if len(s.litbuf) > 0 {
		s.litbuf = append(s.litbuf, lit...)
		lit = s.litbuf
	}

	return string(lit)
}

/*
// getr reads and returns the next Unicode character. It is designed such
// that only a minimal amount of work needs to be done in the common ASCII
// case (a single test to check for both ASCII and end-of-buffer, and one
// test each to check for NUL and to count newlines).
func (s *scanner) getr() rune {
	// unread rune != 0 available
	if r := s.peekr1; r != 0 {
		s.peekr1 = s.peekr2
		s.peekr2 = 0
		if r == '\n' && importpkg == nil {
			lexlineno++
		}
		return r
	}

redo:
	// common case: 7bit ASCII
	if b := s.buf[s.pos]; b < utf8.RuneSelf {
		s.pos++
		if b == 0 {
			// TODO(gri) do we need lineno = lexlineno here?
			Yyerror("illegal NUL byte")
			return 0
		}
		if b == '\n' && importpkg == nil {
			lexlineno++
		}
		return rune(b)
	}

	// uncommon case: not ASCII or not enough bytes
	for s.pos+utf8.UTFMax > s.end && !utf8.FullRune(s.buf[s.pos:s.end]) {
		// not enough bytes: read some more, but first
		// move unread bytes to beginning of buffer
		copy(s.buf[0:], s.buf[s.pos:s.end])
		// read more bytes
		// (an io.Reader must return io.EOF when it reaches
		// the end of what it is reading - simply returning
		// n == 0 will make this loop retry forever; but the
		// error is in the reader implementation in that case)
		// TODO(gri) check for it an return io.ErrNoProgress?
		// (see also bufio.go:666)
		i := s.end - s.pos
		n, err := s.src.Read(s.buf[i : len(s.buf)-1])
		s.pos = 0
		s.end = i + n
		s.buf[s.end] = utf8.RuneSelf // sentinel
		if err != nil {
			if s.end == 0 {
				return EOF
			}
			if err != io.EOF {
				panic(err) // TODO(gri) fix this
			}
			// If err == EOF, we won't be getting more
			// bytes; break to avoid infinite loop. If
			// err is something else, we don't know if
			// we can get more bytes; thus also break.
			break
		}
	}

	// we have at least one byte (excluding sentinel)
	// common case: 7bit ASCII
	if b := s.buf[s.pos]; b < utf8.RuneSelf {
		s.pos++
		if b == 0 {
			// TODO(gri) do we need lineno = lexlineno here?
			Yyerror("illegal NUL byte")
			return 0
		}
		if b == '\n' && importpkg == nil {
			lexlineno++
		}
		return rune(b)
	}

	// uncommon case: not ASCII
	r, w := utf8.DecodeRune(s.buf[s.pos:s.end])
	s.pos += w
	if r == utf8.RuneError && w == 1 {
		lineno = lexlineno
		// The string conversion here makes a copy for passing
		// to fmt.Printf, so that buf itself does not escape and
		// can be allocated on the stack.
		Yyerror("illegal UTF-8 sequence %x", r)
	}

	if r == BOM {
		yyerrorl(int(lexlineno), "Unicode (UTF-8) BOM in middle of file")
		goto redo
	}

	return r
}

// pos returns the position of the most recently read character s.ch.
func (s *Scanner) pos() Offset {
	// TODO(gri) consider replacing lastCharLen with chPos or equivalent
	return Offset(s.srcBufOffset + s.srcPos - s.chLen)
}

func (s *Scanner) startLiteral() {
	s.symBuf = s.symBuf[:0]
	s.symPos = s.srcPos - s.chLen
}

func (s *Scanner) stopLiteral(stripCR bool) string {
	symEnd := s.srcPos - s.chLen

	lit := s.srcBuf[s.symPos:symEnd]
	s.symPos = -1
	if len(s.symBuf) > 0 {
		// part of the symbol text was saved in symBuf: save the rest in
		// symBuf as well and return its content
		s.symBuf = append(s.symBuf, lit...)
		lit = s.symBuf
	}

	if stripCR {
		c := make([]byte, len(lit))
		i := 0
		for _, ch := range lit {
			if ch != '\r' {
				c[i] = ch
				i++
			}
		}
		lit = c[:i]
	}

	return string(lit)
}
*/
