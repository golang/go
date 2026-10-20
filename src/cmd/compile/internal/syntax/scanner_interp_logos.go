// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

// This file holds the loGo/s scanner logic for M6-S1 interpolated
// strings (SPEC Sec 1.4, T-104.4). scanner.go calls into it through one
// hook line only (G3); the rule runs solely in LogosMode, so plain Go
// keeps the exact upstream invalid-character path for `$`.
//
// `$"..."` scans as one _Literal token with LitKind InterpLit when `$`
// joins the opening quote with no gap. The literal Value keeps the full
// source spelling (including `$"` and the closing quote), so the parser
// can split literal parts from `{expr}` operands at exact positions.
// `{{` and `}}` in literal text are literal braces; a lone `}` there is
// an error, and a `{` without a matching `}` is left for the parser,
// which reports `syntax error: unterminated interpolation` at the brace.
// Braces inside an operand nest, with nested strings, runes, raw
// strings, and comments scanned so their braces never count.

// logosInterp scans a `$"..."` interpolated string. The scanner is
// positioned at `$` with the token segment started there. It reports
// false without consuming input (up to a rewind) when `$` does not join
// a quote, so the caller keeps the upstream invalid-character path.
func (s *scanner) logosInterp() bool {
	if s.mode&uint(LogosMode) == 0 || s.ch != '$' {
		return false
	}
	s.nextch()
	if s.ch != '"' {
		nl := s.ch == '\n'
		s.rewind() // back to '$'; the peek consumed no newline itself,
		if nl {    // but rewind counts the peeked one, so undo it.
			s.line--
		}
		return false
	}
	s.nextch() // consume the opening quote
	ok := true
	depth := 0
	for {
		switch c := s.ch; {
		case c < 0:
			if depth > 0 {
				// The parser reports the unmatched brace; the
				// token completes so parsing can continue.
				s.setLit(InterpLit, true)
				return true
			}
			s.errorAtf(0, "string not terminated")
			s.setLit(InterpLit, false)
			return true
		case c == '\n':
			if depth > 0 {
				s.nextch()
				continue
			}
			s.errorf("newline in string")
			s.setLit(InterpLit, false)
			return true
		case depth == 0 && c == '"':
			s.nextch()
			s.setLit(InterpLit, ok)
			return true
		case depth == 0 && c == '\\':
			s.nextch()
			if !s.escape('"') {
				ok = false
			}
		case depth == 0 && c == '{':
			s.nextch()
			if s.ch == '{' {
				s.nextch()
				continue
			}
			depth++
		case depth == 0 && c == '}':
			s.nextch()
			if s.ch == '}' {
				s.nextch()
				continue
			}
			s.errorf("unescaped '}' in interpolated string")
			ok = false
		case c == '{':
			depth++
			s.nextch()
		case c == '}':
			depth--
			s.nextch()
		case c == '"':
			if !s.logosNestedString() {
				ok = false
			}
		case c == '\'':
			if !s.logosNestedRune() {
				ok = false
			}
		case c == '`':
			if !s.logosNestedRaw() {
				ok = false
			}
		case c == '/':
			// logosNestedComment consumed the `/`; on false the
			// operator character now current still needs consuming.
			if !s.logosNestedComment() {
				s.nextch()
			}
		case c == '\\':
			s.nextch()
			if s.ch >= 0 {
				s.nextch()
			}
		default:
			s.nextch()
		}
	}
}

// logosNestedString scans a nested interpreted string inside an
// interpolation operand, mirroring stdString without ending the token.
// It reports whether the string was well-formed.
func (s *scanner) logosNestedString() bool {
	ok := true
	s.nextch()
	for {
		switch c := s.ch; {
		case c == '"':
			s.nextch()
			return ok
		case c == '\\':
			s.nextch()
			if !s.escape('"') {
				ok = false
			}
		case c == '\n':
			s.errorf("newline in string")
			return false
		case c < 0:
			s.errorAtf(0, "string not terminated")
			return false
		default:
			s.nextch()
		}
	}
}

// logosNestedRune scans a nested rune literal inside an interpolation
// operand, mirroring rune without ending the token. It reports whether
// the literal was well-formed.
func (s *scanner) logosNestedRune() bool {
	ok := true
	s.nextch()
	n := 0
	for ; ; n++ {
		switch c := s.ch; {
		case c == '\'':
			if ok {
				if n == 0 {
					s.errorf("empty rune literal or unescaped '")
					ok = false
				} else if n != 1 {
					s.errorAtf(0, "more than one character in rune literal")
					ok = false
				}
			}
			s.nextch()
			return ok
		case c == '\\':
			s.nextch()
			if !s.escape('\'') {
				ok = false
			}
		case c == '\n':
			if ok {
				s.errorf("newline in rune literal")
				ok = false
			}
			return false
		case c < 0:
			if ok {
				s.errorAtf(0, "rune literal not terminated")
				ok = false
			}
			return false
		default:
			s.nextch()
		}
	}
}

// logosNestedRaw scans a nested raw string inside an interpolation
// operand, mirroring rawString without ending the token. It reports
// whether the string was terminated.
func (s *scanner) logosNestedRaw() bool {
	s.nextch()
	for {
		if s.ch == '`' {
			s.nextch()
			return true
		}
		if s.ch < 0 {
			s.errorAtf(0, "string not terminated")
			return false
		}
		s.nextch()
	}
}

// logosNestedComment scans a comment starting at `/` inside an
// interpolation operand so braces inside comments never count. It
// reports true when it consumed a comment, false when `/` starts no
// comment and the caller must scan it as an operator.
func (s *scanner) logosNestedComment() bool {
	s.nextch()
	if s.ch == '/' {
		for s.ch != '\n' && s.ch >= 0 {
			s.nextch()
		}
		return true
	}
	if s.ch == '*' {
		s.nextch()
		for {
			if s.ch < 0 {
				s.errorAtf(0, "comment not terminated")
				return true
			}
			if s.ch == '*' {
				s.nextch()
				if s.ch == '/' {
					s.nextch()
					return true
				}
				continue
			}
			s.nextch()
		}
	}
	return false
}
