// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import (
	"fmt"
	"io"
	"strings"
	"unicode"
	"unicode/utf8"
)

type scanner struct {
	source
	nlsemi bool // if set '\n' and EOF translate to ';'

	// current token, valid after calling next()
	pos, line int
	tok       token
	lit       string   // valid if tok is _Name or _Literal
	op        Operator // valid if tok is _Operator, _AssignOp, or _IncOp
	prec      int      // valid if tok is _Operator, _AssignOp, or _IncOp

	pragmas []Pragma
}

func (s *scanner) init(src io.Reader) {
	s.source.init(src)
	s.nlsemi = false
}

func (s *scanner) next() {
	nlsemi := s.nlsemi
	s.nlsemi = false

redo:
	// skip white space
	c := s.getr()
	for c == ' ' || c == '\t' || c == '\n' && !nlsemi || c == '\r' {
		c = s.getr()
	}

	// token start
	s.pos, s.line = s.source.pos(), s.source.line

	if isLetter(c) || c >= utf8.RuneSelf && unicode.IsLetter(c) {
		s.ident()
		return
	}

	switch c {
	case -1:
		s.tok = _EOF

	case '\n':
		// ';' is before the '\n'
		s.pos--
		s.line--
		s.tok = _Semi

	case '0', '1', '2', '3', '4', '5', '6', '7', '8', '9':
		s.number(c)
		s.nlsemi = true
		s.tok = _Literal

	case '"':
		s.stdString()
		s.nlsemi = true
		s.tok = _Literal

	case '`':
		s.rawString()
		s.nlsemi = true
		s.tok = _Literal

	case '\'':
		s.rune()
		s.nlsemi = true
		s.tok = _Literal

	case '(':
		s.tok = _Lparen

	case '[':
		s.tok = _Lbrack

	case '{':
		s.tok = _Lbrace

	case ',':
		s.tok = _Comma

	case ';':
		s.tok = _Semi

	case ')':
		s.nlsemi = true
		s.tok = _Rparen

	case ']':
		s.nlsemi = true
		s.tok = _Rbrack

	case '}':
		s.nlsemi = true
		s.tok = _Rbrace

	case ':':
		if s.getr() == '=' {
			s.tok = _Define
			break
		}
		s.ungetr()
		s.tok = _Colon

	case '.':
		c = s.getr()
		if isDigit(c) {
			s.ungetr()
			s.source.r0-- // make sure '.' is part of literal (line cannot have changed)
			s.number('.')
			s.nlsemi = true
			s.tok = _Literal
			break
		}
		if c == '.' {
			c = s.getr()
			if c == '.' {
				s.tok = _DotDotDot
				break
			}
			s.ungetr()
			s.source.r0-- // make next ungetr work (line cannot have changed)
		}
		s.ungetr()
		s.tok = _Dot

	case '+':
		s.op, s.prec = Add, precAdd
		c = s.getr()
		if c != '+' {
			goto assignop
		}
		s.nlsemi = true
		s.tok = _IncOp

	case '-':
		s.op, s.prec = Sub, precAdd
		c = s.getr()
		if c != '-' {
			goto assignop
		}
		s.nlsemi = true
		s.tok = _IncOp

	case '*':
		s.op, s.prec = Mul, precMul
		// don't goto assignop - want _Star token
		if s.getr() == '=' {
			s.tok = _AssignOp
			return
		}
		s.ungetr()
		s.tok = _Star

	case '/':
		c = s.getr()
		if c == '/' {
			s.lineComment()
			goto redo
		}
		if c == '*' {
			s.fullComment()
			if s.source.line > s.line && nlsemi {
				// A multi-line comment acts like a newline;
				// it translates to a ';' if nlsemi is set.
				s.tok = _Semi
				break
			}
			goto redo
		}
		s.op, s.prec = Div, precMul
		goto assignop

	case '%':
		s.op, s.prec = Rem, precMul
		c = s.getr()
		goto assignop

	case '&':
		c = s.getr()
		if c == '&' {
			s.op, s.prec = AndAnd, precAndAnd
			s.tok = _Operator
			break
		}
		s.op, s.prec = And, precMul
		if c == '^' {
			s.op = AndNot
			c = s.getr()
		}
		goto assignop

	case '|':
		c = s.getr()
		if c == '|' {
			s.op, s.prec = OrOr, precOrOr
			s.tok = _Operator
			break
		}
		s.op, s.prec = Or, precAdd
		goto assignop

	case '~':
		panic("bitwise complement operator is ^")
		fallthrough

	case '^':
		s.op, s.prec = Xor, precAdd
		c = s.getr()
		goto assignop

	case '<':
		c = s.getr()
		if c == '=' {
			s.op, s.prec = Leq, precCmp
			s.tok = _Operator
			break
		}
		if c == '<' {
			s.op, s.prec = Shl, precMul
			c = s.getr()
			goto assignop
		}
		if c == '-' {
			s.tok = _Arrow
			break
		}
		s.ungetr()
		s.op, s.prec = Lss, precCmp
		s.tok = _Operator

	case '>':
		c = s.getr()
		if c == '=' {
			s.op, s.prec = Geq, precCmp
			s.tok = _Operator
			break
		}
		if c == '>' {
			s.op, s.prec = Shr, precMul
			c = s.getr()
			goto assignop
		}
		s.ungetr()
		s.op, s.prec = Gtr, precCmp
		s.tok = _Operator

	case '=':
		if s.getr() == '=' {
			s.op, s.prec = Eql, precCmp
			s.tok = _Operator
			break
		}
		s.ungetr()
		s.tok = _Assign

	case '!':
		if s.getr() == '=' {
			s.op, s.prec = Neq, precCmp
			s.tok = _Operator
			break
		}
		s.ungetr()
		s.op, s.prec = Not, 0
		s.tok = _Operator

	default:
		s.tok = 0
		fmt.Printf("invalid rune %q\n", c)
		panic("invalid rune")
		goto redo
	}

	return

assignop:
	if c == '=' {
		s.tok = _AssignOp
		return
	}
	s.ungetr()
	s.tok = _Operator
}

func isLetter(c rune) bool {
	return 'a' <= c && c <= 'z' || 'A' <= c && c <= 'Z' || c == '_'
}

func isDigit(c rune) bool {
	return '0' <= c && c <= '9'
}

func (s *scanner) ident() {
	s.startLit()

	// accelerate common case (7bit ASCII)
	c := s.getr()
	for isLetter(c) || isDigit(c) {
		c = s.getr()
	}

	// general case
	if c >= utf8.RuneSelf {
		for unicode.IsLetter(c) || c == '_' || unicode.IsDigit(c) {
			c = s.getr()
		}
	}
	s.ungetr()

	lit := s.stopLit()

	// possibly a keyword
	if len(lit) >= 2 {
		if tok := keywordMap[hash(lit)]; tok != 0 && strbyteseql(tokstrings[tok], lit) {
			s.nlsemi = contains(1<<_Break|1<<_Continue|1<<_Fallthrough|1<<_Return, tok)
			s.tok = tok
			return
		}
	}

	s.nlsemi = true
	s.tok = _Name
	s.lit = string(lit)
}

// hash is a perfect hash function for keywords.
// It assumes that s has at least length 2.
func hash(s []byte) uint {
	return (uint(s[0])<<4 ^ uint(s[1]) + uint(len(s))) & uint(len(keywordMap)-1)
}

func strbyteseql(s string, b []byte) bool {
	if len(s) == len(b) {
		for i, b := range b {
			if s[i] != b {
				return false
			}
		}
		return true
	}
	return false
}

var keywordMap [1 << 6]token // size must be power of two

func init() {
	// populate keywordMap
	for tok := _Break; tok <= _Var; tok++ {
		h := hash([]byte(tokstrings[tok]))
		if keywordMap[h] != 0 {
			panic("imperfect hash")
		}
		keywordMap[h] = tok
	}
}

func (s *scanner) number(c rune) {
	s.startLit()

	if c != '.' {
		if c == '0' {
			c = s.getr()
			if c == 'x' || c == 'X' {
				// hex
				c = s.getr()
				hasDigit := false
				for isDigit(c) || 'a' <= c && c <= 'f' || 'A' <= c && c <= 'F' {
					c = s.getr()
					hasDigit = true
				}
				if !hasDigit {
					panic("malformed hex constant")
				}
				s.ungetr()
				s.lit = string(s.stopLit())
				return
			}

			// decimal 0, octal, or float
			has8or9 := false
			for isDigit(c) {
				if c > '7' {
					has8or9 = true
				}
				c = s.getr()
			}
			if c != '.' && c != 'e' && c != 'E' && c != 'i' {
				// octal
				if has8or9 {
					panic("malformed octal constant")
				}
				s.ungetr()
				s.lit = string(s.stopLit())
				return
			}

		} else {
			// decimal or float
			for isDigit(c) {
				c = s.getr()
			}
		}
	}

	// float
	if c == '.' {
		c = s.getr()
		for isDigit(c) {
			c = s.getr()
		}
	}

	// exponent
	if c == 'e' || c == 'E' {
		c = s.getr()
		if c == '-' || c == '+' {
			c = s.getr()
		}
		if !isDigit(c) {
			panic("malformed floating-point constant exponent")
		}
		for isDigit(c) {
			c = s.getr()
		}
	}

	// complex
	if c != 'i' {
		s.ungetr() // not complex
	}

	s.lit = string(s.stopLit())
}

func (s *scanner) stdString() {
	s.startLit()
	for {
		r := s.getr()
		if r == '\\' && !s.escape('"') {
			panic(0)
		}
		if r == '"' {
			break
		}
		if r < 0 {
			panic("string not terminated")
		}
	}
	s.lit = string(s.stopLit())
}

func (s *scanner) rawString() {
	s.startLit()
	for {
		r := s.getr()
		if r == '`' {
			break
		}
		if r < 0 {
			panic("string not terminated")
		}
		// TODO(gri) deal with CRs (or don't?)
	}
	s.lit = string(s.stopLit())
}

func (s *scanner) rune() {
	s.startLit()
	r := s.getr()
	if r == '\\' && !s.escape('\'') {
		panic(0)
	}
	c := s.getr()
	if c != '\'' {
		panic(c)
	}
	s.lit = string(s.stopLit())
}

func (s *scanner) lineComment() {
	// recognize pragmas
	var prefix string
	r := s.getr()
	switch r {
	case 'g':
		prefix = "go:"
	case 'l':
		prefix = "line "
	default:
		goto skip
	}

	s.startLit()
	for _, m := range prefix {
		if r != m {
			s.stopLit()
			goto skip
		}
		r = s.getr()
	}

	for r >= 0 {
		if r == '\n' {
			s.ungetr()
			break
		}
		r = s.getr()
	}
	s.pragmas = append(s.pragmas, Pragma{
		Line: s.line,
		Text: strings.TrimSuffix(string(s.stopLit()), "\r"),
	})
	return

skip:
	// consume line
	for r != '\n' && r >= 0 {
		r = s.getr()
	}
	s.ungetr() // don't consume '\n' - needed for nlsemi logic
}

func (s *scanner) fullComment() {
	for {
		r := s.getr()
		for r == '*' {
			r = s.getr()
			if r == '/' {
				return
			}
		}
		if r < 0 {
			panic("comment not terminated")
		}
	}
}

func (s *scanner) escape(quote rune) bool {
	var n int
	var base, max uint32

	c := s.getr()
	switch c {
	case 'a', 'b', 'f', 'n', 'r', 't', 'v', '\\', quote:
		return true
	case '0', '1', '2', '3', '4', '5', '6', '7':
		n, base, max = 3, 8, 255
	case 'x':
		c = s.getr()
		n, base, max = 2, 16, 255
	case 'u':
		c = s.getr()
		n, base, max = 4, 16, unicode.MaxRune
	case 'U':
		c = s.getr()
		n, base, max = 8, 16, unicode.MaxRune
	default:
		var msg string
		if c >= 0 {
			msg = "unknown escape sequence"
		} else {
			msg = "escape sequence not terminated"
		}
		panic(msg)
		return false
	}

	var x uint32
loop:
	for ; n > 0; n-- {
		d := base
		switch {
		case isDigit(c):
			d = uint32(c) - '0'
		case 'a' <= c && c <= 'f':
			d = uint32(c) - ('a' - 10)
		case 'A' <= c && c <= 'F':
			d = uint32(c) - ('A' - 10)
		}
		if d >= base {
			var msg string
			if c >= 0 {
				msg = fmt.Sprintf("illegal character %#U in escape sequence", c)
			} else {
				msg = "escape sequence not terminated"
			}
			panic(msg)
			break loop
		}
		// d < base
		x = x*base + d
		c = s.getr()
	}
	s.ungetr()

	if x > max || 0xD800 <= x && x < 0xE000 /* surrogate range */ {
		panic("escape sequence is invalid Unicode code point")
		return false
	}

	return true
}
