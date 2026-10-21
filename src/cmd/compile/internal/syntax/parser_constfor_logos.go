// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

// This file holds the loGo/s parser logic for M7-S2 (`const for` over
// concrete types) and M7-S3 (comptime generics) (SPEC Sec 2.5, T-205).
// parser.go calls into it through hook lines only (G3); every rule below
// runs solely in LogosMode, so plain Go parses exactly as upstream.

// logosConstForStmt recognizes a `const for` statement and returns it.
// It returns nil unless LogosMode is on and the word after `const` is
// `for`, leaving every ordinary `const` declaration on the upstream
// path. The peek is a read-only byte scan: blanks and comments are
// skipped, the buffer is filled as needed, and nothing is consumed, so
// positions stay exact when the rule does not fire.
func (p *parser) logosConstForStmt() Stmt {
	if p.mode&LogosMode == 0 || !p.peekConstWord() {
		return nil
	}
	pos := p.pos()
	p.next()
	p.want(_For)
	var index, field *Name
	first := p.name()
	if p.tok == _Comma {
		p.next()
		index = first
		field = p.name()
	} else {
		field = first
	}
	p.want(_Define)
	p.want(_Range)
	s := new(ConstForStmt)
	s.pos = pos
	s.Index = index
	s.Field = field
	outer := p.xnest
	p.xnest = -1
	s.Range = p.expr()
	p.xnest = outer
	s.Body = p.blockStmt("const for clause")
	return s
}

// peekConstWord reports whether the next source word after the current
// `const` token is `for`. It mirrors the logosLeadingDot fill
// discipline: offsets stay valid across fills, and the scan never
// consumes.
func (p *parser) peekConstWord() bool {
	q := p.r - p.chw
	get := func() (byte, bool) {
		for q >= p.e {
			if p.ioerr != nil {
				return 0, false
			}
			q -= p.fillPeek()
		}
		c := p.buf[q]
		q++
		return c, true
	}
	for {
		c, ok := get()
		if !ok {
			return false
		}
		if c == ' ' || c == '\t' || c == '\r' || c == '\n' {
			continue
		}
		if c == '/' {
			c2, ok := get()
			if !ok {
				return false
			}
			if c2 == '/' {
				for {
					c3, ok := get()
					if !ok || c3 == '\n' {
						break
					}
				}
				continue
			}
			if c2 == '*' {
				for {
					c3, ok := get()
					if !ok {
						return false
					}
					if c3 == '*' {
						c4, ok := get()
						if !ok {
							return false
						}
						if c4 == '/' {
							break
						}
						q--
					}
				}
				continue
			}
			return false
		}
		if c != 'f' {
			return false
		}
		var word [3]byte
		word[0] = c
		for i := 1; i < 3; i++ {
			c, ok := get()
			if !ok {
				return false
			}
			word[i] = c
		}
		if string(word[:]) != "for" {
			return false
		}
		c2, ok2 := get()
		if !ok2 {
			return true
		}
		return !isWordChar(c2)
	}
}

// isWordChar reports whether c continues an identifier word.
func isWordChar(c byte) bool {
	return c == '_' || 'a' <= c && c <= 'z' || 'A' <= c && c <= 'Z' || '0' <= c && c <= '9'
}

// logosFieldProj recognizes a `.[f]` projection on x and returns it. It
// returns nil unless LogosMode is on and the token after `.` is `[`,
// leaving every ordinary selector and the error paths on the upstream
// way. Only a bare name is accepted between the brackets (SPEC Sec 2.5
// FieldProj); anything else is a syntax error.
func (p *parser) logosFieldProj(x Expr, pos Pos) Expr {
	if p.mode&LogosMode == 0 || p.tok != _Lbrack {
		return nil
	}
	p.next()
	name := p.name()
	rbrack := p.pos()
	p.want(_Rbrack)
	t := new(FieldProjExpr)
	t.pos = pos
	t.X = x
	t.Field = name
	t.Rbrack = rbrack
	return t
}
