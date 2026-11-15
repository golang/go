// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import (
	"unicode"
	"unicode/utf8"
)

func (s *scanner) logosFatArrow() bool {
	if s.mode&uint(LogosMode) == 0 || s.ch != '>' {
		return false
	}
	s.nextch()
	s.tok = _FatArrow
	return true
}

func (s *scanner) fillPeek() int {
	b := s.r
	if s.b >= 0 {
		b = s.b
	}
	s.fill()
	return b
}

func (s *scanner) logosLeadingDot() bool {
	if s.mode&uint(LogosMode) == 0 || s.ch != '\n' || s.tok == _Question {
		return false
	}
	r := s.r
	for {
		for r < s.e {
			c := s.buf[r]
			if c != ' ' && c != '\t' && c != '\n' && c != '\r' {
				goto found
			}
			r++
		}
		if s.ioerr != nil {
			return false
		}
		r -= s.fillPeek()
	}
found:
	if s.buf[r] != '.' {
		return false
	}
	r++
	if r >= s.e {
		if s.ioerr != nil {
			return false
		}
		r -= s.fillPeek()
	}
	c := s.buf[r]
	if c == '_' || isLetter(rune(c)) {
		return true
	}
	if c < utf8.RuneSelf {
		return false
	}
	for !utf8.FullRune(s.buf[r:s.e]) {
		if s.ioerr != nil {
			return false
		}
		r -= s.fillPeek()
	}
	ru, _ := utf8.DecodeRune(s.buf[r:s.e])
	return unicode.IsLetter(ru)
}
