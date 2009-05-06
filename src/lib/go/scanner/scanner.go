// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// A scanner for Go source text. Takes a []byte as source which can
// then be tokenized through repeated calls to the Scan function.
// For a sample use of a scanner, see the implementation of Tokenize.
//
package scanner

import (
	"go/token";
	"strconv";
	"unicode";
	"utf8";
)


// An implementation of an ErrorHandler may be provided to the Scanner.
// If a syntax error is encountered and a handler was installed, Error
// is called with a position and an error message. The position points
// to the beginning of the offending token.
//
type ErrorHandler interface {
	Error(pos token.Position, msg string);
}


// A Scanner holds the scanner's internal state while processing
// a given text.  It can be allocated as part of another data
// structure but must be initialized via Init before use. For
// a sample use, see the implementation of Tokenize.
//
type Scanner struct {
	// immutable state
	src []byte;  // source
	err ErrorHandler;  // error reporting; or nil
	scan_comments bool;  // if set, comments are reported as tokens

	// scanning state
	pos token.Position;  // previous reading position (position before ch)
	offset int;  // current reading offset (position after ch)
	ch int;  // one char look-ahead

	// public state - ok to modify
	ErrorCount int;  // number of errors encountered
}


// Read the next Unicode char into S.ch.
// S.ch < 0 means end-of-file.
//
func (S *Scanner) next() {
	if S.offset < len(S.src) {
		S.pos.Offset = S.offset;
		S.pos.Column++;
		r, w := int(S.src[S.offset]), 1;
		switch {
		case r == '\n':
			S.pos.Line++;
			S.pos.Column = 0;
		case r >= 0x80:
			// not ASCII
			r, w = utf8.DecodeRune(S.src[S.offset : len(S.src)]);
		}
		S.offset += w;
		S.ch = r;
	} else {
		S.pos.Offset = len(S.src);
		S.ch = -1;  // eof
	}
}


// Init prepares the scanner S to tokenize the text src. Calls to Scan
// will use the error handler err if they encounter a syntax error and
// err is not nil. Also, for each error encountered, the Scanner field
// ErrorCount is incremented by one. The boolean scan_comments specifies
// whether comments should be recognized and returned by Scan as COMMENT
// tokens. If scan_comments is false, they are treated as white space and
// ignored.
//
func (S *Scanner) Init(src []byte, err ErrorHandler, scan_comments bool) {
	// Explicitly initialize all fields since a scanner may be reused.
	S.src = src;
	S.err = err;
	S.scan_comments = scan_comments;
	S.pos = token.Position{0, 1, 0};
	S.offset = 0;
	S.ErrorCount = 0;
	S.next();
}


func charString(ch int) string {
	var s string;
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
	default  : s = string(ch);
	}
	return "'" + s + "' (U+" + strconv.Itob(ch, 16) + ")";
}


func (S *Scanner) error(pos token.Position, msg string) {
	if S.err != nil {
		S.err.Error(pos, msg);
	}
	S.ErrorCount++;
}


func (S *Scanner) expect(ch int) {
	if S.ch != ch {
		S.error(S.pos, "expected " + charString(ch) + ", found " + charString(S.ch));
	}
	S.next();  // always make progress
}


func (S *Scanner) scanComment(pos token.Position) {
	// first '/' already consumed

	if S.ch == '/' {
		//-style comment
		for S.ch >= 0 {
			S.next();
			if S.ch == '\n' {
				S.next();  // '\n' belongs to the comment
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

	S.error(pos, "comment not terminated");
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


func (S *Scanner) scanIdentifier() token.Token {
	pos := S.pos.Offset;
	for isLetter(S.ch) || isDigit(S.ch) {
		S.next();
	}
	return token.Lookup(S.src[pos : S.pos.Offset]);
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


func (S *Scanner) scanNumber(seen_decimal_point bool) token.Token {
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
		S.error(S.pos, "illegal char escape");
	}
}


func (S *Scanner) scanEscape(quote int) {
	pos := S.pos;
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
		S.error(pos, "illegal char escape");
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


func (S *Scanner) scanString(pos token.Position) {
	// '"' already consumed

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
}


func (S *Scanner) scanRawString(pos token.Position) {
	// '`' already consumed

	for S.ch != '`' {
		ch := S.ch;
		S.next();
		if ch == '\n' || ch < 0 {
			S.error(pos, "string not terminated");
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

func (S *Scanner) switch2(tok0, tok1 token.Token) token.Token {
	if S.ch == '=' {
		S.next();
		return tok1;
	}
	return tok0;
}


func (S *Scanner) switch3(tok0, tok1 token.Token, ch2 int, tok2 token.Token) token.Token {
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


func (S *Scanner) switch4(tok0, tok1 token.Token, ch2 int, tok2, tok3 token.Token) token.Token {
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


// Scan scans the next token and returns the token position pos,
// the token tok, and the literal text lit corresponding to the
// token. The source end is indicated by token.EOF.
//
// For more tolerant parsing, Scan will return a valid token if
// possible even if a syntax error was encountered. Thus, even
// if the resulting token sequence contains no illegal tokens,
// a client may not assume that no error occurred. Instead it
// must check the scanner's ErrorCount or the number of calls
// of the error handler, if there was one installed.
//
func (S *Scanner) Scan() (pos token.Position, tok token.Token, lit []byte) {
scan_again:
	// skip white space
	for S.ch == ' ' || S.ch == '\t' || S.ch == '\n' || S.ch == '\r' {
		S.next();
	}

	// current token start
	pos, tok = S.pos, token.ILLEGAL;

	// determine token value
	switch ch := S.ch; {
	case isLetter(ch):
		tok = S.scanIdentifier();
	case digitVal(ch) < 10:
		tok = S.scanNumber(false);
	default:
		S.next();  // always make progress
		switch ch {
		case -1  : tok = token.EOF;
		case '"' : tok = token.STRING; S.scanString(pos);
		case '\'': tok = token.CHAR; S.scanChar();
		case '`' : tok = token.STRING; S.scanRawString(pos);
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
				S.scanComment(pos);
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
		case '&':
			if S.ch == '^' {
				S.next();
				tok = S.switch2(token.AND_NOT, token.AND_NOT_ASSIGN);
			} else {
				tok = S.switch3(token.AND, token.AND_ASSIGN, '&', token.LAND);
			}
		case '|': tok = S.switch3(token.OR, token.OR_ASSIGN, '|', token.LOR);
		default: S.error(pos, "illegal character " + charString(ch));
		}
	}

	return pos, tok, S.src[pos.Offset : S.pos.Offset];
}


// Tokenize calls a function f with the token position, token value, and token
// text for each token in the source src. The other parameters have the same
// meaning as for the Init function. Tokenize keeps scanning until f returns
// false (usually when the token value is token.EOF). The result is the number
// of errors encountered.
//
func Tokenize(src []byte, err ErrorHandler, scan_comments bool, f func (pos token.Position, tok token.Token, lit []byte) bool) int {
	var s Scanner;
	s.Init(src, err, scan_comments);
	for f(s.Scan()) {
		// action happens in f
	}
	return s.ErrorCount;
}
