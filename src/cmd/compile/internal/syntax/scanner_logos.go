// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import (
	"unicode"
	"unicode/utf8"
)

// This file holds the loGo/s scanner logic for M2-S5 (`=>`) and M6-S8
// (leading-dot continuation). scanner.go calls into it through one hook
// line per rule only (G3).

// logosFatArrow completes the `=>` token. It is called right after the
// scanner consumed '=' and found no second '='. In LogosMode, a '>'
// that joins the '=' with no gap makes the pair one _FatArrow token
// (SPEC Sec 5.1, T-104.6); the token ends no operand, so a following
// newline yields no ';'. Otherwise it reports false and leaves the
// scanner untouched, so plain Go keeps scanning `=` followed by `>`
// exactly as upstream and `= >` with a gap keeps both tokens in
// LogosMode.
func (s *scanner) logosFatArrow() bool {
	if s.mode&uint(LogosMode) == 0 || s.ch != '>' {
		return false
	}
	s.nextch()
	s.tok = _FatArrow
	return true
}

// fillPeek reads more source bytes for a lookahead and reports the
// coordinate shift of the buffer compaction, so a peek offset stays
// valid across the fill. The base mirrors source.fill: the active
// segment start when one is open, otherwise the read position.
func (s *scanner) fillPeek() int {
	b := s.r
	if s.b >= 0 {
		b = s.b
	}
	s.fill()
	return b
}

// logosLeadingDot reports whether a newline at the current position
// starts a leading-dot continuation (SPEC Sec 5.3, T-104.6): in
// LogosMode, a newline followed by spaces, tabs, carriage returns, or
// further newlines and then `.` plus a letter or `_` inserts no
// semicolon. It peeks at the buffered source without consuming, filling
// the buffer as needed, so positions stay exact when the rule does not
// fire. A newline after `?` still ends the statement (SPEC T-104.1
// takes precedence, pinned by TestLogosQuestionDotScan), and a `.`
// followed by a digit (`x\n.5`), a second dot, or anything else keeps
// the semicolon; plain Go never reaches it.
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
	// A multibyte identifier start: decode the rune, filling for the
	// remaining bytes, and require a letter the way the scanner's own
	// identifier entry does. Continuation bytes, non-letters, and
	// invalid encodings keep the semicolon.
	for !utf8.FullRune(s.buf[r:s.e]) {
		if s.ioerr != nil {
			return false
		}
		r -= s.fillPeek()
	}
	ru, _ := utf8.DecodeRune(s.buf[r:s.e])
	return unicode.IsLetter(ru)
}
