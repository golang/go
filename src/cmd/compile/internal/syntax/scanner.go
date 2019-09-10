// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements scanner, a lexical tokenizer for
// Go source. After initialization, consecutive calls of
// next advance the scanner one token at a time.
//
// This file, source.go, and tokens.go are self-contained
// (go tool compile scanner.go source.go tokens.go compiles)
// and thus could be made into its own package.

package syntax

import (
	"fmt"
	"io"
	"unicode"
	"unicode/utf8"
)

// The mode flags below control which comments are reported
// by calling the error handler. If no flag is set, comments
// are ignored.
const (
	comments   uint = 1 << iota // call handler for all comments
	directives                  // call handler for directives only
)

type scanner struct {
	source
	mode   uint
	nlsemi bool // if set '\n' and EOF translate to ';'

	// current token, valid after calling next()
	line, col uint
	tok       token
	lit       string   // valid if tok is _Name, _Literal, or _Semi ("semicolon", "newline", or "EOF"); may be malformed if bad is true
	bad       bool     // valid if tok is _Literal, true if a syntax error occurred, lit may be malformed
	kind      LitKind  // valid if tok is _Literal
	op        Operator // valid if tok is _Operator, _AssignOp, or _IncOp
	prec      int      // valid if tok is _Operator, _AssignOp, or _IncOp
}

func (s *scanner) init(src io.Reader, errh func(line, col uint, msg string), mode uint) {
	s.source.init(src, errh)
	s.mode = mode
	s.nlsemi = false
}

// errorf reports an error at the most recently read character position.
func (s *scanner) errorf(format string, args ...interface{}) {
	s.bad = true
	s.error(fmt.Sprintf(format, args...))
}

// errorAtf reports an error at a byte column offset relative to the current token start.
func (s *scanner) errorAtf(offset int, format string, args ...interface{}) {
	s.bad = true
	s.errh(s.line, s.col+uint(offset), fmt.Sprintf(format, args...))
}

// next advances the scanner by reading the next token.
//
// If a read, source encoding, or lexical error occurs, next calls
// the installed error handler with the respective error position
// and message. The error message is guaranteed to be non-empty and
// never starts with a '/'. The error handler must exist.
//
// If the scanner mode includes the comments flag and a comment
// (including comments containing directives) is encountered, the
// error handler is also called with each comment position and text
// (including opening /* or // and closing */, but without a newline
// at the end of line comments). Comment text always starts with a /
// which can be used to distinguish these handler calls from errors.
//
// If the scanner mode includes the directives (but not the comments)
// flag, only comments containing a //line, /*line, or //go: directive
// are reported, in the same way as regular comments. Directives in
// //-style comments are only recognized if they are at the beginning
// of a line.
//
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
	s.line, s.col = s.source.line0, s.source.col0

	if isLetter(c) || c >= utf8.RuneSelf && s.isIdentRune(c, true) {
		s.ident()
		return
	}

	switch c {
	case -1:
		if nlsemi {
			s.lit = "EOF"
			s.tok = _Semi
			break
		}
		s.tok = _EOF

	case '\n':
		s.lit = "newline"
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
		s.lit = "semicolon"
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
		if isDecimal(c) {
			s.ungetr()
			s.unread(1) // correct position of '.' (needed by startLit in number)
			s.number('.')
			break
		}
		if c == '.' {
			c = s.getr()
			if c == '.' {
				s.tok = _DotDotDot
				break
			}
			s.unread(1)
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
				s.lit = "newline"
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
		s.errorf("invalid character %#U", c)
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
	return 'a' <= lower(c) && lower(c) <= 'z' || c == '_'
}

func (s *scanner) ident() {
	s.startLit()

	// accelerate common case (7bit ASCII)
	c := s.getr()
	for isLetter(c) || isDecimal(c) {
		c = s.getr()
	}

	// general case
	if c >= utf8.RuneSelf {
		for s.isIdentRune(c, false) {
			c = s.getr()
		}
	}
	s.ungetr()

	lit := s.stopLit()

	// possibly a keyword
	if len(lit) >= 2 {
		if tok := keywordMap[hash(lit)]; tok != 0 && tokStrFast(tok) == string(lit) {
			s.nlsemi = contains(1<<_Break|1<<_Continue|1<<_Fallthrough|1<<_Return, tok)
			s.tok = tok
			return
		}
	}

	s.nlsemi = true
	s.lit = string(lit)
	s.tok = _Name
}

// tokStrFast is a faster version of token.String, which assumes that tok
// is one of the valid tokens - and can thus skip bounds checks.
func tokStrFast(tok token) string {
	return _token_name[_token_index[tok-1]:_token_index[tok]]
}

func (s *scanner) isIdentRune(c rune, first bool) bool {
	switch {
	case unicode.IsLetter(c) || c == '_':
		// ok
	case unicode.IsDigit(c):
		if first {
			s.errorf("identifier cannot begin with digit %#U", c)
		}
	case c >= utf8.RuneSelf:
		s.errorf("invalid identifier character %#U", c)
	default:
		return false
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
		h := hash([]byte(tok.String()))
		if keywordMap[h] != 0 {
			panic("imperfect hash")
		}
		keywordMap[h] = tok
	}
}

func lower(c rune) rune     { return ('a' - 'A') | c } // returns lower-case c iff c is ASCII letter
func isDecimal(c rune) bool { return '0' <= c && c <= '9' }
func isHex(c rune) bool     { return '0' <= c && c <= '9' || 'a' <= lower(c) && lower(c) <= 'f' }

// digits accepts the sequence { digit | '_' } starting with c0.
// If base <= 10, digits accepts any decimal digit but records
// the index (relative to the literal start) of a digit >= base
// in *invalid, if *invalid < 0.
// digits returns the first rune that is not part of the sequence
// anymore, and a bitset describing whether the sequence contained
// digits (bit 0 is set), or separators '_' (bit 1 is set).
func (s *scanner) digits(c0 rune, base int, invalid *int) (c rune, digsep int) {
	c = c0
	if base <= 10 {
		max := rune('0' + base)
		for isDecimal(c) || c == '_' {
			ds := 1
			if c == '_' {
				ds = 2
			} else if c >= max && *invalid < 0 {
				*invalid = int(s.col0 - s.col) // record invalid rune index
			}
			digsep |= ds
			c = s.getr()
		}
	} else {
		for isHex(c) || c == '_' {
			ds := 1
			if c == '_' {
				ds = 2
			}
			digsep |= ds
			c = s.getr()
		}
	}
	return
}

func (s *scanner) number(c rune) {
	s.startLit()
	s.bad = false

	base := 10        // number base
	prefix := rune(0) // one of 0 (decimal), '0' (0-octal), 'x', 'o', or 'b'
	digsep := 0       // bit 0: digit present, bit 1: '_' present
	invalid := -1     // index of invalid digit in literal, or < 0

	// integer part
	var ds int
	if c != '.' {
		s.kind = IntLit
		if c == '0' {
			c = s.getr()
			switch lower(c) {
			case 'x':
				c = s.getr()
				base, prefix = 16, 'x'
			case 'o':
				c = s.getr()
				base, prefix = 8, 'o'
			case 'b':
				c = s.getr()
				base, prefix = 2, 'b'
			default:
				base, prefix = 8, '0'
				digsep = 1 // leading 0
			}
		}
		c, ds = s.digits(c, base, &invalid)
		digsep |= ds
	}

	// fractional part
	if c == '.' {
		s.kind = FloatLit
		if prefix == 'o' || prefix == 'b' {
			s.errorf("invalid radix point in %s", litname(prefix))
		}
		c, ds = s.digits(s.getr(), base, &invalid)
		digsep |= ds
	}

	if digsep&1 == 0 && !s.bad {
		s.errorf("%s has no digits", litname(prefix))
	}

	// exponent
	if e := lower(c); e == 'e' || e == 'p' {
		if !s.bad {
			switch {
			case e == 'e' && prefix != 0 && prefix != '0':
				s.errorf("%q exponent requires decimal mantissa", c)
			case e == 'p' && prefix != 'x':
				s.errorf("%q exponent requires hexadecimal mantissa", c)
			}
		}
		c = s.getr()
		s.kind = FloatLit
		if c == '+' || c == '-' {
			c = s.getr()
		}
		c, ds = s.digits(c, 10, nil)
		digsep |= ds
		if ds&1 == 0 && !s.bad {
			s.errorf("exponent has no digits")
		}
	} else if prefix == 'x' && s.kind == FloatLit && !s.bad {
		s.errorf("hexadecimal mantissa requires a 'p' exponent")
	}

	// suffix 'i'
	if c == 'i' {
		s.kind = ImagLit
		c = s.getr()
	}
	s.ungetr()

	s.nlsemi = true
	s.lit = string(s.stopLit())
	s.tok = _Literal

	if s.kind == IntLit && invalid >= 0 && !s.bad {
		s.errorAtf(invalid, "invalid digit %q in %s", s.lit[invalid], litname(prefix))
	}

	if digsep&2 != 0 && !s.bad {
		if i := invalidSep(s.lit); i >= 0 {
			s.errorAtf(i, "'_' must separate successive digits")
		}
	}
}

func litname(prefix rune) string {
	switch prefix {
	case 'x':
		return "hexadecimal literal"
	case 'o', '0':
		return "octal literal"
	case 'b':
		return "binary literal"
	}
	return "decimal literal"
}

// invalidSep returns the index of the first invalid separator in x, or -1.
func invalidSep(x string) int {
	x1 := ' ' // prefix char, we only care if it's 'x'
	d := '.'  // digit, one of '_', '0' (a digit), or '.' (anything else)
	i := 0

	// a prefix counts as a digit
	if len(x) >= 2 && x[0] == '0' {
		x1 = lower(rune(x[1]))
		if x1 == 'x' || x1 == 'o' || x1 == 'b' {
			d = '0'
			i = 2
		}
	}

	// mantissa and exponent
	for ; i < len(x); i++ {
		p := d // previous digit
		d = rune(x[i])
		switch {
		case d == '_':
			if p != '0' {
				return i
			}
		case isDecimal(d) || x1 == 'x' && isHex(d):
			d = '0'
		default:
			if p == '_' {
				return i - 1
			}
			d = '.'
		}
	}
	if d == '_' {
		return len(x) - 1
	}

	return -1
}

func (s *scanner) rune() {
	s.startLit()
	s.bad = false

	n := 0
	for ; ; n++ {
		r := s.getr()
		if r == '\'' {
			break
		}
		if r == '\\' {
			s.escape('\'')
			continue
		}
		if r == '\n' {
			s.ungetr() // assume newline is not part of literal
			if !s.bad {
				s.errorf("newline in character literal")
			}
			break
		}
		if r < 0 {
			if !s.bad {
				s.errorAtf(0, "invalid character literal (missing closing ')")
			}
			break
		}
	}

	if !s.bad {
		if n == 0 {
			s.errorf("empty character literal or unescaped ' in character literal")
		} else if n != 1 {
			s.errorAtf(0, "invalid character literal (more than one character)")
		}
	}

	s.nlsemi = true
	s.lit = string(s.stopLit())
	s.kind = RuneLit
	s.tok = _Literal
}

func (s *scanner) stdString() {
	s.startLit()
	s.bad = false

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
			s.errorf("newline in string")
			break
		}
		if r < 0 {
			s.errorAtf(0, "string not terminated")
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
	s.bad = false

	for {
		r := s.getr()
		if r == '`' {
			break
		}
		if r < 0 {
			s.errorAtf(0, "string not terminated")
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

func (s *scanner) comment(text string) {
	s.errh(s.line, s.col, text)
}

func (s *scanner) skipLine(r rune) {
	for r >= 0 {
		if r == '\n' {
			s.ungetr() // don't consume '\n' - needed for nlsemi logic
			break
		}
		r = s.getr()
	}
}

func (s *scanner) lineComment() {
	r := s.getr()

	if s.mode&comments != 0 {
		s.startLit()
		s.skipLine(r)
		s.comment("//" + string(s.stopLit()))
		return
	}

	// directives must start at the beginning of the line (s.col == colbase)
	if s.mode&directives == 0 || s.col != colbase || (r != 'g' && r != 'l') {
		s.skipLine(r)
		return
	}

	// recognize go: or line directives
	prefix := "go:"
	if r == 'l' {
		prefix = "line "
	}
	for _, m := range prefix {
		if r != m {
			s.skipLine(r)
			return
		}
		r = s.getr()
	}

	// directive text
	s.startLit()
	s.skipLine(r)
	s.comment("//" + prefix + string(s.stopLit()))
}

func (s *scanner) skipComment(r rune) bool {
	for r >= 0 {
		for r == '*' {
			r = s.getr()
			if r == '/' {
				return true
			}
		}
		r = s.getr()
	}
	s.errorAtf(0, "comment not terminated")
	return false
}

func (s *scanner) fullComment() {
	r := s.getr()

	if s.mode&comments != 0 {
		s.startLit()
		if s.skipComment(r) {
			s.comment("/*" + string(s.stopLit()))
		} else {
			s.killLit() // not a complete comment - ignore
		}
		return
	}

	if s.mode&directives == 0 || r != 'l' {
		s.skipComment(r)
		return
	}

	// recognize line directive
	const prefix = "line "
	for _, m := range prefix {
		if r != m {
			s.skipComment(r)
			return
		}
		r = s.getr()
	}

	// directive text
	s.startLit()
	if s.skipComment(r) {
		s.comment("/*" + prefix + string(s.stopLit()))
	} else {
		s.killLit() // not a complete comment - ignore
	}
}

func (s *scanner) escape(quote rune) {
	var n int
	var base, max uint32

	c := s.getr()
	switch c {
	case 'a', 'b', 'f', 'n', 'r', 't', 'v', '\\', quote:
		return
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
			return // complain in caller about EOF
		}
		s.errorf("unknown escape sequence")
		return
	}

	var x uint32
	for i := n; i > 0; i-- {
		d := base
		switch {
		case isDecimal(c):
			d = uint32(c) - '0'
		case 'a' <= lower(c) && lower(c) <= 'f':
			d = uint32(lower(c)) - ('a' - 10)
		}
		if d >= base {
			if c < 0 {
				return // complain in caller about EOF
			}
			kind := "hex"
			if base == 8 {
				kind = "octal"
			}
			s.errorf("non-%s character in escape sequence: %c", kind, c)
			s.ungetr()
			return
		}
		// d < base
		x = x*base + d
		c = s.getr()
	}
	s.ungetr()

	if x > max && base == 8 {
		s.errorf("octal escape value > 255: %d", x)
		return
	}

	if x > max || 0xD800 <= x && x < 0xE000 /* surrogate range */ {
		s.errorf("escape sequence is invalid Unicode code point %#U", x)
	}
}
