// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import (
	"fmt"
	"io"
	"unicode/utf8"
)

// buf [...read...|...|...unread...|s|...free...]
//         ^      ^   ^            ^
//         |      |   |            |
//        suf     r0  r            w

type source struct {
	src  io.Reader
	errh ErrorHandler

	// source buffer
	buf         [4 << 10]byte
	offs        int   // source offset of buf
	r0, r, w    int   // previous/current read and write buf positions, excluding sentinel
	line0, line int   // previous/current line
	err         error // pending io error

	// literal buffer
	lit []byte // literal prefix
	suf int    // literal suffix; suf >= 0 means we are scanning a literal
}

func (s *source) init(src io.Reader, errh ErrorHandler) {
	s.src = src
	s.errh = errh

	s.buf[0] = utf8.RuneSelf // terminate with sentinel
	s.offs = 0
	s.r0, s.r, s.w = 0, 0, 0
	s.line0, s.line = 1, 1
	s.err = nil

	s.lit = s.lit[:0]
	s.suf = -1
}

func (s *source) error(msg string) {
	s.error_at(s.pos(), s.line, msg)
}

func (s *source) error_at(pos, line int, msg string) {
	if s.errh != nil {
		s.errh(pos, line, msg)
		return
	}
	panic(fmt.Sprintf("%d: %s", line, msg))
}

func (s *source) pos() int {
	return s.offs + s.r
}

func (s *source) ungetr() {
	s.r, s.line = s.r0, s.line0
}

func (s *source) getr() rune {
	for {
		s.r0, s.line0 = s.r, s.line

		// common case: ASCII and enough bytes
		if b := s.buf[s.r]; b < utf8.RuneSelf {
			s.r++
			if b == 0 {
				s.error("invalid NUL character")
				continue
			}
			if b == '\n' {
				s.line++
			}
			return rune(b)
		}

		// uncommon case: not ASCII or not enough bytes
		r, w := utf8.DecodeRune(s.buf[s.r:s.w]) // optimistically assume valid rune
		if r != utf8.RuneError || w > 1 {
			s.r += w
			// BOM's are only allowed as the first character in a file
			const BOM = 0xfeff
			if r == BOM && s.r0 > 0 { // s.r0 is always > 0 after 1st character (fill will set it to 1)
				s.error("invalid BOM in the middle of the file")
				continue
			}
			return r
		}

		if w == 0 && s.err != nil {
			if s.err != io.EOF {
				s.error(s.err.Error())
			}
			return -1
		}

		if w == 1 && (s.r+utf8.UTFMax <= s.w || utf8.FullRune(s.buf[s.r:s.w])) {
			s.r++
			s.error("invalid UTF-8 encoding")
			continue
		}

		s.fill()
	}
}

func (s *source) fill() {
	// Slide unread bytes to beginning but preserve last read char
	// (for one ungetr call) plus one extra byte (for a 2nd ungetr
	// call, only for ".." character sequence).
	if s.r0 > 1 {
		// save literal prefix, if any
		// (We see at most one ungetr call while reading
		// a literal, so make sure s.r0 remains in buf.)
		if s.suf >= 0 {
			s.lit = append(s.lit, s.buf[s.suf:s.r0]...)
			s.suf = 1 // == s.r0 after slide below
		}
		s.offs += s.r0 - 1
		r := s.r - s.r0 + 1 // last read char plus one byte
		s.w = r + copy(s.buf[r:], s.buf[s.r:s.w])
		s.r = r
		s.r0 = 1
	}

	// read more data: try a limited number of times
	for i := 100; i > 0; i-- {
		n, err := s.src.Read(s.buf[s.w : len(s.buf)-1]) // -1 to leave space for sentinel
		if n < 0 {
			s.error("negative read")
		}
		s.w += n
		if n > 0 || err != nil {
			s.buf[s.w] = utf8.RuneSelf // sentinel
			if err != nil {
				s.err = err
			}
			return
		}
	}

	s.error("no progress")
}

func (s *source) startLit() {
	s.suf = s.r0
	s.lit = s.lit[:0] // reuse lit
}

func (s *source) stopLit() []byte {
	lit := s.buf[s.suf:s.r]
	if len(s.lit) > 0 {
		lit = append(s.lit, lit...)
	}
	s.suf = -1 // no pending literal
	return lit
}
