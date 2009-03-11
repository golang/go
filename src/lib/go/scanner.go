// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// A scanner for Go source text. Takes a []byte as source which can
// then be tokenized through repeated calls to the Scan function.
//
// Sample use:
//
//	import "token"
//	import "scanner"
//
//	func tokenize(src []byte) {
//		var s scanner.Scanner;
//		s.Init(src, nil /* no error handler */, false /* ignore comments */);
//		for {
//			pos, tok, lit := s.Scan();
//			if tok == Scanner.EOF {
//				return;
//			}
//			println(pos, token.TokenString(tok), string(lit));
//		}
//	}
//
package scanner

import (
	"utf8";
	"unicode";
	"strconv";
	"token";
)


// Source locations are represented by a Location value.
type Location struct {
	Pos int;  // byte position in source
	Line int;  // line count, starting at 1
	Col int;  // column, starting at 1 (character count)
}


// An implementation of an ErrorHandler must be provided to the Scanner.
// If a syntax error is encountered, Error is called with a location and
// an error message. The location points at the beginning of the offending
// token.
//
type ErrorHandler interface {
	Error(loc Location, msg string);
}


// A Scanner holds the scanner's internal state while processing
// a given text.  It can be allocated as part of another data
// structure but must be initialized via Init before use.
// See also the package comment for a sample use.
//
type Scanner struct {
	// immutable state
	src []byte;  // source
	err ErrorHandler;  // error reporting
	scan_comments bool;  // if set, comments are reported as tokens

	// scanning state
	loc Location;  // location of ch
	pos int;  // current reading position (position after ch)
	ch int;  // one char look-ahead
}


// Read the next Unicode char into S.ch.
// S.ch < 0 means end-of-file.
func (S *Scanner) next() {
	if S.pos < len(S.src) {
		S.loc.Pos = S.pos;
		S.loc.Col++;
		r, w := int(S.src[S.pos]), 1;
		switch {
		case r == '\n':
			S.loc.Line++;
			S.loc.Col = 1;
		case r >= 0x80:
			// not ASCII
			r, w = utf8.DecodeRune(S.src[S.pos : len(S.src)]);
		}
		S.pos += w;
		S.ch = r;
	} else {
		S.loc.Pos = len(S.src);
		S.ch = -1;  // eof
	}
}


// Init prepares the scanner S to tokenize the text src. Calls to Scan
// will use the error handler err if they encounter a syntax error. The boolean
// scan_comments specifies whether newline characters and comments should be
// recognized and returned by Scan as token.COMMENT. If scan_comments is false,
// they are treated as white space and ignored.
//
func (S *Scanner) Init(src []byte, err ErrorHandler, scan_comments bool) {
	S.src = src;
	S.err = err;
	S.scan_comments = scan_comments;
	S.loc.Line = 1;
	S.next();
}


func charString(ch int) string {
	s := string(ch);
	switch ch {
	case '\a': s = `\a`;
	case '\b': s = `\b`;
	case '\f': s = `\f`;
	case '\n': s = `\n`;
	case '\r': s = `\r`;
	case '\t': s = `\t`;
	case '\v': s = `\v`;
	case '\\': s = `\\`;
	case '\'': s = `\'`;
	}
	return "'" + s + "' (U+" + strconv.Itob(ch, 16) + ")";
}


func (S *Scanner) error(loc Location, msg string) {
	S.err.Error(loc, msg);
}


func (S *Scanner) expect(ch int) {
	if S.ch != ch {
		S.error(S.loc, "expected " + charString(ch) + ", found " + charString(S.ch));
	}
	S.next();  // always make progress
}


func (S *Scanner) skipWhitespace() {
	for {
		switch S.ch {
		case '\t', '\r', ' ':
			// nothing to do
		case '\n':
			if S.scan_comments {
				return;
			}
		default:
			return;
		}
		S.next();
	}
	panic("UNREACHABLE");
}


func (S *Scanner) scanComment(loc Location) {
	// first '/' already consumed

	if S.ch == '/' {
		//-style comment
		for S.ch >= 0 {
			S.next();
			if S.ch == '\n' {
				// '\n' terminates comment but we do not include
				// it in the comment (otherwise we don't see the
				// start of a newline in skipWhitespace()).
				return;
			}
		}

	} else {
		/*-style comment */
		S.expect('*');
		for S.ch >= 0 {
			ch := S.ch;
			S.next();
			if ch == '*' && S.ch == '/' {
				S.next();
				return;
			}
		}
	}

	S.error(loc, "comment not terminated");
}


func isLetter(ch int) bool {
	return
		'a' <= ch && ch <= 'z' ||
		'A' <= ch && ch <= 'Z' ||
		ch == '_' ||
		ch >= 0x80 && unicode.IsLetter(ch);
}


func isDigit(ch int) bool {
	return
		'0' <= ch && ch <= '9' ||
		ch >= 0x80 && unicode.IsDecimalDigit(ch);
}


func (S *Scanner) scanIdentifier() int {
	pos := S.loc.Pos;
	for isLetter(S.ch) || isDigit(S.ch) {
		S.next();
	}
	return token.Lookup(S.src[pos : S.loc.Pos]);
}


func digitVal(ch int) int {
	switch {
	case '0' <= ch && ch <= '9': return ch - '0';
	case 'a' <= ch && ch <= 'f': return ch - 'a' + 10;
	case 'A' <= ch && ch <= 'F': return ch - 'A' + 10;
	}
	return 16;  // larger than any legal digit val
}


func (S *Scanner) scanMantissa(base int) {
	for digitVal(S.ch) < base {
		S.next();
	}
}


func (S *Scanner) scanNumber(seen_decimal_point bool) int {
	tok := token.INT;

	if seen_decimal_point {
		tok = token.FLOAT;
		S.scanMantissa(10);
		goto exponent;
	}

	if S.ch == '0' {
		// int or float
		S.next();
		if S.ch == 'x' || S.ch == 'X' {
			// hexadecimal int
			S.next();
			S.scanMantissa(16);
		} else {
			// octal int or float
			S.scanMantissa(8);
			if digitVal(S.ch) < 10 || S.ch == '.' || S.ch == 'e' || S.ch == 'E' {
				// float
				tok = token.FLOAT;
				goto mantissa;
			}
			// octal int
		}
		goto exit;
	}

mantissa:
	// decimal int or float
	S.scanMantissa(10);

	if S.ch == '.' {
		// float
		tok = token.FLOAT;
		S.next();
		S.scanMantissa(10)
	}

exponent:
	if S.ch == 'e' || S.ch == 'E' {
		// float
		tok = token.FLOAT;
		S.next();
		if S.ch == '-' || S.ch == '+' {
			S.next();
		}
		S.scanMantissa(10);
	}

exit:
	return tok;
}


func (S *Scanner) scanDigits(base, length int) {
	for length > 0 && digitVal(S.ch) < base {
		S.next();
		length--;
	}
	if length > 0 {
		S.error(S.loc, "illegal char escape");
	}
}


func (S *Scanner) scanEscape(quote int) {
	loc := S.loc;
	ch := S.ch;
	S.next();
	switch ch {
	case 'a', 'b', 'f', 'n', 'r', 't', 'v', '\\', quote:
		// nothing to do
	case '0', '1', '2', '3', '4', '5', '6', '7':
		S.scanDigits(8, 3 - 1);  // 1 char read already
	case 'x':
		S.scanDigits(16, 2);
	case 'u':
		S.scanDigits(16, 4);
	case 'U':
		S.scanDigits(16, 8);
	default:
		S.error(loc, "illegal char escape");
	}
}


func (S *Scanner) scanChar() {
	// '\'' already consumed

	ch := S.ch;
	S.next();
	if ch == '\\' {
		S.scanEscape('\'');
	}

	S.expect('\'');
}


func (S *Scanner) scanString(loc Location) {
	// '"' already consumed

	for S.ch != '"' {
		ch := S.ch;
		S.next();
		if ch == '\n' || ch < 0 {
			S.error(loc, "string not terminated");
			break;
		}
		if ch == '\\' {
			S.scanEscape('"');
		}
	}

	S.next();
}


func (S *Scanner) scanRawString(loc Location) {
	// '`' already consumed

	for S.ch != '`' {
		ch := S.ch;
		S.next();
		if ch == '\n' || ch < 0 {
			S.error(loc, "string not terminated");
			break;
		}
	}

	S.next();
}


// Helper functions for scanning multi-byte tokens such as >> += >>= .
// Different routines recognize different length tok_i based on matches
// of ch_i. If a token ends in '=', the result is tok1 or tok3
// respectively. Otherwise, the result is tok0 if there was no other
// matching character, or tok2 if the matching character was ch2.

func (S *Scanner) switch2(tok0, tok1 int) int {
	if S.ch == '=' {
		S.next();
		return tok1;
	}
	return tok0;
}


func (S *Scanner) switch3(tok0, tok1, ch2, tok2 int) int {
	if S.ch == '=' {
		S.next();
		return tok1;
	}
	if S.ch == ch2 {
		S.next();
		return tok2;
	}
	return tok0;
}


func (S *Scanner) switch4(tok0, tok1, ch2, tok2, tok3 int) int {
	if S.ch == '=' {
		S.next();
		return tok1;
	}
	if S.ch == ch2 {
		S.next();
		if S.ch == '=' {
			S.next();
			return tok3;
		}
		return tok2;
	}
	return tok0;
}


// Scan scans the next token and returns the token location loc,
// the token tok, and the literal text lit corresponding to the
// token.
//
func (S *Scanner) Scan() (loc Location, tok int, lit []byte) {
scan_again:
	S.skipWhitespace();

	loc, tok = S.loc, token.ILLEGAL;

	switch ch := S.ch; {
	case isLetter(ch):
		tok = S.scanIdentifier();
	case digitVal(ch) < 10:
		tok = S.scanNumber(false);
	default:
		S.next();  // always make progress
		switch ch {
		case -1  : tok = token.EOF;
		case '\n': tok = token.COMMENT;
		case '"' : tok = token.STRING; S.scanString(loc);
		case '\'': tok = token.CHAR; S.scanChar();
		case '`' : tok = token.STRING; S.scanRawString(loc);
		case ':' : tok = S.switch2(token.COLON, token.DEFINE);
		case '.' :
			if digitVal(S.ch) < 10 {
				tok = S.scanNumber(true);
			} else if S.ch == '.' {
				S.next();
				if S.ch == '.' {
					S.next();
					tok = token.ELLIPSIS;
				}
			} else {
				tok = token.PERIOD;
			}
		case ',': tok = token.COMMA;
		case ';': tok = token.SEMICOLON;
		case '(': tok = token.LPAREN;
		case ')': tok = token.RPAREN;
		case '[': tok = token.LBRACK;
		case ']': tok = token.RBRACK;
		case '{': tok = token.LBRACE;
		case '}': tok = token.RBRACE;
		case '+': tok = S.switch3(token.ADD, token.ADD_ASSIGN, '+', token.INC);
		case '-': tok = S.switch3(token.SUB, token.SUB_ASSIGN, '-', token.DEC);
		case '*': tok = S.switch2(token.MUL, token.MUL_ASSIGN);
		case '/':
			if S.ch == '/' || S.ch == '*' {
				S.scanComment(loc);
				tok = token.COMMENT;
				if !S.scan_comments {
					goto scan_again;
				}
			} else {
				tok = S.switch2(token.QUO, token.QUO_ASSIGN);
			}
		case '%': tok = S.switch2(token.REM, token.REM_ASSIGN);
		case '^': tok = S.switch2(token.XOR, token.XOR_ASSIGN);
		case '<':
			if S.ch == '-' {
				S.next();
				tok = token.ARROW;
			} else {
				tok = S.switch4(token.LSS, token.LEQ, '<', token.SHL, token.SHL_ASSIGN);
			}
		case '>': tok = S.switch4(token.GTR, token.GEQ, '>', token.SHR, token.SHR_ASSIGN);
		case '=': tok = S.switch2(token.ASSIGN, token.EQL);
		case '!': tok = S.switch2(token.NOT, token.NEQ);
		case '&': tok = S.switch3(token.AND, token.AND_ASSIGN, '&', token.LAND);
		case '|': tok = S.switch3(token.OR, token.OR_ASSIGN, '|', token.LOR);
		default: S.error(loc, "illegal character " + charString(ch));
		}
	}

	return loc, tok, S.src[loc.Pos : S.loc.Pos];
}
