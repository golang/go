// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package scanner

// A Go scanner. Takes a []byte as source which can then be
// tokenized through repeated calls to the Scan() function.
//
// Sample use:
//
//  import "token"
//  import "scanner"
//
//	func tokenize(src []byte) {
//		var s scanner.Scanner;
//		s.Init(src, nil, false);
//		for {
//			pos, tok, lit := s.Scan();
//			if tok == Scanner.EOF {
//				return;
//			}
//			println(pos, token.TokenString(tok), string(lit));
//		}
//	}

import (
	"utf8";
	"unicode";
	"strconv";
	"token";
)

type ErrorHandler interface {
	Error(pos int, msg string);
}


type Scanner struct {
	// setup
	src []byte;  // source
	err ErrorHandler;
	scan_comments bool;

	// scanning
	pos int;  // current reading position
	ch int;  // one char look-ahead
	chpos int;  // position of ch
}


func is_letter(ch int) bool {
	return
		'a' <= ch && ch <= 'z' || 'A' <= ch && ch <= 'Z' ||  // common case
		ch == '_' || unicode.IsLetter(ch);
}


func digit_val(ch int) int {
	// TODO spec permits other Unicode digits as well
	if '0' <= ch && ch <= '9' {
		return ch - '0';
	}
	if 'a' <= ch && ch <= 'f' {
		return ch - 'a' + 10;
	}
	if 'A' <= ch && ch <= 'F' {
		return ch - 'A' + 10;
	}
	return 16;  // larger than any legal digit val
}


// Read the next Unicode char into S.ch.
// S.ch < 0 means end-of-file.
func (S *Scanner) next() {
	if S.pos < len(S.src) {
		// assume ascii
		r, w := int(S.src[S.pos]), 1;
		if r >= 0x80 {
			// not ascii
			r, w = utf8.DecodeRune(S.src[S.pos : len(S.src)]);
		}
		S.ch = r;
		S.chpos = S.pos;
		S.pos += w;
	} else {
		S.ch = -1;  // eof
		S.chpos = len(S.src);
	}
}


// Initialize the scanner.
//
// The error handler (err) is called when an illegal token is encountered.
// If scan_comments is set to true, newline characters ('\n') and comments
// are recognized as token.COMMENT, otherwise they are treated as white
// space and ignored.

func (S *Scanner) Init(src []byte, err ErrorHandler, scan_comments bool) {
	S.src = src;
	S.err = err;
	S.scan_comments = scan_comments;
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


func (S *Scanner) error(pos int, msg string) {
	S.err.Error(pos, msg);
}


func (S *Scanner) expect(ch int) {
	if S.ch != ch {
		S.error(S.chpos, "expected " + charString(ch) + ", found " + charString(S.ch));
	}
	S.next();  // make always progress
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


func (S *Scanner) scanComment() []byte {
	// first '/' already consumed
	pos := S.chpos - 1;

	if S.ch == '/' {
		//-style comment
		for S.ch >= 0 {
			S.next();
			if S.ch == '\n' {
				// '\n' terminates comment but we do not include
				// it in the comment (otherwise we don't see the
				// start of a newline in skipWhitespace()).
				goto exit;
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
				goto exit;
			}
		}
	}

	S.error(pos, "comment not terminated");

exit:
	return S.src[pos : S.chpos];
}


func (S *Scanner) scanIdentifier() (tok int, lit []byte) {
	pos := S.chpos;
	for is_letter(S.ch) || digit_val(S.ch) < 10 {
		S.next();
	}
	lit = S.src[pos : S.chpos];
	return token.Lookup(lit), lit;
}


func (S *Scanner) scanMantissa(base int) {
	for digit_val(S.ch) < base {
		S.next();
	}
}


func (S *Scanner) scanNumber(seen_decimal_point bool) (tok int, lit []byte) {
	pos := S.chpos;
	tok = token.INT;

	if seen_decimal_point {
		tok = token.FLOAT;
		pos--;  // '.' is one byte
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
			if digit_val(S.ch) < 10 || S.ch == '.' || S.ch == 'e' || S.ch == 'E' {
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
	return tok, S.src[pos : S.chpos];
}


func (S *Scanner) scanDigits(n int, base int) {
	for digit_val(S.ch) < base {
		S.next();
		n--;
	}
	if n > 0 {
		S.error(S.chpos, "illegal char escape");
	}
}


func (S *Scanner) scanEscape(quote int) {
	ch := S.ch;
	pos := S.chpos;
	S.next();
	switch ch {
	case 'a', 'b', 'f', 'n', 'r', 't', 'v', '\\', quote:
		// nothing to do
	case '0', '1', '2', '3', '4', '5', '6', '7':
		S.scanDigits(3 - 1, 8);  // 1 char read already
	case 'x':
		S.scanDigits(2, 16);
	case 'u':
		S.scanDigits(4, 16);
	case 'U':
		S.scanDigits(8, 16);
	default:
		S.error(pos, "illegal char escape");
	}
}


func (S *Scanner) scanChar() []byte {
	// '\'' already consumed

	pos := S.chpos - 1;
	ch := S.ch;
	S.next();
	if ch == '\\' {
		S.scanEscape('\'');
	}

	S.expect('\'');
	return S.src[pos : S.chpos];
}


func (S *Scanner) scanString() []byte {
	// '"' already consumed

	pos := S.chpos - 1;
	for S.ch != '"' {
		ch := S.ch;
		S.next();
		if ch == '\n' || ch < 0 {
			S.error(pos, "string not terminated");
			break;
		}
		if ch == '\\' {
			S.scanEscape('"');
		}
	}

	S.next();
	return S.src[pos : S.chpos];
}


func (S *Scanner) scanRawString() []byte {
	// '`' already consumed

	pos := S.chpos - 1;
	for S.ch != '`' {
		ch := S.ch;
		S.next();
		if ch == '\n' || ch < 0 {
			S.error(pos, "string not terminated");
			break;
		}
	}

	S.next();
	return S.src[pos : S.chpos];
}


func (S *Scanner) select2(tok0, tok1 int) int {
	if S.ch == '=' {
		S.next();
		return tok1;
	}
	return tok0;
}


func (S *Scanner) select3(tok0, tok1, ch2, tok2 int) int {
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


func (S *Scanner) select4(tok0, tok1, ch2, tok2, tok3 int) int {
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


// Scans the next token. Returns the token byte position in the source,
// its token value, and the corresponding literal text if the token is
// an identifier or basic type literals (token.IsLiteral(tok) == true).

func (S *Scanner) Scan() (pos, tok int, lit []byte) {
loop:
	S.skipWhitespace();

	pos, tok = S.chpos, token.ILLEGAL;

	switch ch := S.ch; {
	case is_letter(ch): tok, lit = S.scanIdentifier();
	case digit_val(ch) < 10: tok, lit = S.scanNumber(false);
	default:
		S.next();  // always make progress
		switch ch {
		case -1: tok = token.EOF;
		case '\n': tok, lit = token.COMMENT, []byte{'\n'};
		case '"': tok, lit = token.STRING, S.scanString();
		case '\'': tok, lit = token.CHAR, S.scanChar();
		case '`': tok, lit = token.STRING, S.scanRawString();
		case ':': tok = S.select2(token.COLON, token.DEFINE);
		case '.':
			if digit_val(S.ch) < 10 {
				tok, lit = S.scanNumber(true);
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
		case '+': tok = S.select3(token.ADD, token.ADD_ASSIGN, '+', token.INC);
		case '-': tok = S.select3(token.SUB, token.SUB_ASSIGN, '-', token.DEC);
		case '*': tok = S.select2(token.MUL, token.MUL_ASSIGN);
		case '/':
			if S.ch == '/' || S.ch == '*' {
				tok, lit = token.COMMENT, S.scanComment();
				if !S.scan_comments {
					goto loop;
				}
			} else {
				tok = S.select2(token.QUO, token.QUO_ASSIGN);
			}
		case '%': tok = S.select2(token.REM, token.REM_ASSIGN);
		case '^': tok = S.select2(token.XOR, token.XOR_ASSIGN);
		case '<':
			if S.ch == '-' {
				S.next();
				tok = token.ARROW;
			} else {
				tok = S.select4(token.LSS, token.LEQ, '<', token.SHL, token.SHL_ASSIGN);
			}
		case '>': tok = S.select4(token.GTR, token.GEQ, '>', token.SHR, token.SHR_ASSIGN);
		case '=': tok = S.select2(token.ASSIGN, token.EQL);
		case '!': tok = S.select2(token.NOT, token.NEQ);
		case '&': tok = S.select3(token.AND, token.AND_ASSIGN, '&', token.LAND);
		case '|': tok = S.select3(token.OR, token.OR_ASSIGN, '|', token.LOR);
		default:
			S.error(pos, "illegal character " + charString(ch));
		}
	}

	return pos, tok, lit;
}
