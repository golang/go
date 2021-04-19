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
	pragma Pragma

	// current token, valid after calling next()
	pos, line int
	tok       token
	lit       string   // valid if tok is _Name or _Literal
	kind      LitKind  // valid if tok is _Literal
	op        Operator // valid if tok is _Operator, _AssignOp, or _IncOp
	prec      int      // valid if tok is _Operator, _AssignOp, or _IncOp

	pragh PragmaHandler
}

func (s *scanner) init(src io.Reader, errh ErrorHandler, pragh PragmaHandler) {
	s.source.init(src, errh)
	s.nlsemi = false
	s.pragh = pragh
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
	s.pos, s.line = s.source.pos0(), s.source.line0

	if isLetter(c) || c >= utf8.RuneSelf && (unicode.IsLetter(c) || s.isCompatRune(c, true)) {
		s.ident()
		return
	}

	switch c {
	case -1:
		if nlsemi {
			s.tok = _Semi
			break
		}
		s.tok = _EOF

	case '\n':
		s.tok = _Semi

	case '0', '1', '2', '3', '4', '5', '6', '7', '8', '9':
		s.number(c)

	case '"':
		s.stdString()

	case '`':
		s.rawString()

	case '\'':
		s.rune()

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
			break
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
		s.error("bitwise complement operator is ^")
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
		s.error(fmt.Sprintf("illegal character %#U", c))
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
		for unicode.IsLetter(c) || c == '_' || unicode.IsDigit(c) || s.isCompatRune(c, false) {
			c = s.getr()
		}
	}
	s.ungetr()

	lit := s.stopLit()

	// possibly a keyword
	if len(lit) >= 2 {
		if tok := keywordMap[hash(lit)]; tok != 0 && tokstrings[tok] == string(lit) {
			s.nlsemi = contains(1<<_Break|1<<_Continue|1<<_Fallthrough|1<<_Return, tok)
			s.tok = tok
			return
		}
	}

	s.nlsemi = true
	s.lit = string(lit)
	s.tok = _Name
}

func (s *scanner) isCompatRune(c rune, start bool) bool {
	if !gcCompat || c < utf8.RuneSelf {
		return false
	}
	if start && unicode.IsNumber(c) {
		s.error(fmt.Sprintf("identifier cannot begin with digit %#U", c))
	} else {
		s.error(fmt.Sprintf("invalid identifier character %#U", c))
	}
	return true
}

// hash is a perfect hash function for keywords.
// It assumes that s has at least length 2.
func hash(s []byte) uint {
	return (uint(s[0])<<4 ^ uint(s[1]) + uint(len(s))) & uint(len(keywordMap)-1)
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
		s.kind = IntLit // until proven otherwise
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
					s.error("malformed hex constant")
				}
				goto done
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
					s.error("malformed octal constant")
				}
				goto done
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
		s.kind = FloatLit
		c = s.getr()
		for isDigit(c) {
			c = s.getr()
		}
	}

	// exponent
	if c == 'e' || c == 'E' {
		s.kind = FloatLit
		c = s.getr()
		if c == '-' || c == '+' {
			c = s.getr()
		}
		if !isDigit(c) {
			s.error("malformed floating-point constant exponent")
		}
		for isDigit(c) {
			c = s.getr()
		}
	}

	// complex
	if c == 'i' {
		s.kind = ImagLit
		s.getr()
	}

done:
	s.ungetr()
	s.nlsemi = true
	s.lit = string(s.stopLit())
	s.tok = _Literal
}

func (s *scanner) stdString() {
	s.startLit()

	for {
		r := s.getr()
		if r == '"' {
			break
		}
		if r == '\\' {
			s.escape('"')
			continue
		}
		if r == '\n' {
			s.ungetr() // assume newline is not part of literal
			s.error("newline in string")
			break
		}
		if r < 0 {
			s.error_at(s.pos, s.line, "string not terminated")
			break
		}
	}

	s.nlsemi = true
	s.lit = string(s.stopLit())
	s.kind = StringLit
	s.tok = _Literal
}

func (s *scanner) rawString() {
	s.startLit()

	for {
		r := s.getr()
		if r == '`' {
			break
		}
		if r < 0 {
			s.error_at(s.pos, s.line, "string not terminated")
			break
		}
	}
	// We leave CRs in the string since they are part of the
	// literal (even though they are not part of the literal
	// value).

	s.nlsemi = true
	s.lit = string(s.stopLit())
	s.kind = StringLit
	s.tok = _Literal
}

func (s *scanner) rune() {
	s.startLit()

	r := s.getr()
	ok := false
	if r == '\'' {
		s.error("empty character literal or unescaped ' in character literal")
	} else if r == '\n' {
		s.ungetr() // assume newline is not part of literal
		s.error("newline in character literal")
	} else {
		ok = true
		if r == '\\' {
			ok = s.escape('\'')
		}
	}

	r = s.getr()
	if r != '\'' {
		// only report error if we're ok so far
		if ok {
			s.error("missing '")
		}
		s.ungetr()
	}

	s.nlsemi = true
	s.lit = string(s.stopLit())
	s.kind = RuneLit
	s.tok = _Literal
}

func (s *scanner) lineComment() {
	// recognize pragmas
	var prefix string
	r := s.getr()
	if s.pragh == nil {
		goto skip
	}

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
	s.pragma |= s.pragh(0, s.line, strings.TrimSuffix(string(s.stopLit()), "\r"))
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
			s.error_at(s.pos, s.line, "comment not terminated")
			return
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
		if c < 0 {
			return true // complain in caller about EOF
		}
		s.error("unknown escape sequence")
		return false
	}

	var x uint32
	for i := n; i > 0; i-- {
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
			if c < 0 {
				return true // complain in caller about EOF
			}
			if gcCompat {
				name := "hex"
				if base == 8 {
					name = "octal"
				}
				s.error(fmt.Sprintf("non-%s character in escape sequence: %c", name, c))
			} else {
				if c != quote {
					s.error(fmt.Sprintf("illegal character %#U in escape sequence", c))
				} else {
					s.error("escape sequence incomplete")
				}
			}
			s.ungetr()
			return false
		}
		// d < base
		x = x*base + d
		c = s.getr()
	}
	s.ungetr()

	if x > max && base == 8 {
		s.error(fmt.Sprintf("octal escape value > 255: %d", x))
		return false
	}

	if x > max || 0xD800 <= x && x < 0xE000 /* surrogate range */ {
		s.error("escape sequence is invalid Unicode code point")
		return false
	}

	return true
}
