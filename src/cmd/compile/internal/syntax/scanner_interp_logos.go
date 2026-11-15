// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

func (s *scanner) logosInterp() bool {
	if s.mode&uint(LogosMode) == 0 || s.ch != '$' {
		return false
	}
	s.nextch()
	if s.ch != '"' {
		nl := s.ch == '\n'
		s.rewind()
		if nl {
			s.line--
		}
		return false
	}
	s.nextch()
	ok := true
	depth := 0
	for {
		switch c := s.ch; {
		case c < 0:
			if depth > 0 {
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
