// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package scanner implements a scanner for Go source text. Takes a []byte as
// source which can then be tokenized through repeated calls to the Scan
// function. Typical use:
//
//	var s scanner.Scanner
//	fset := token.NewFileSet()  // position information is relative to fset
//      file := fset.AddFile(filename, fset.Base(), len(src))  // register file
//	s.Init(file, src, nil /* no error handler */, 0)
//	for {
//		pos, tok, lit := s.Scan()
//		if tok == token.EOF {
//			break
//		}
//		// do something here with pos, tok, and lit
//	}
//
package scanner

import (
	"bytes"
	"fmt"
	"go/token"
	"path/filepath"
	"strconv"
	"unicode"
	"unicode/utf8"
)

// A Scanner holds the scanner's internal state while processing
// a given text.  It can be allocated as part of another data
// structure but must be initialized via Init before use.
//
type Scanner struct {
	// immutable state
	file *token.File  // source file handle
	dir  string       // directory portion of file.Name()
	src  []byte       // source
	err  ErrorHandler // error reporting; or nil
	mode uint         // scanning mode

	// scanning state
	ch         rune // current character
	offset     int  // character offset
	rdOffset   int  // reading offset (position after current character)
	lineOffset int  // current line offset
	insertSemi bool // insert a semicolon before next newline

	// public state - ok to modify
	ErrorCount int // number of errors encountered
}

// Read the next Unicode char into S.ch.
// S.ch < 0 means end-of-file.
//
func (S *Scanner) next() {
	if S.rdOffset < len(S.src) {
		S.offset = S.rdOffset
		if S.ch == '\n' {
			S.lineOffset = S.offset
			S.file.AddLine(S.offset)
		}
		r, w := rune(S.src[S.rdOffset]), 1
		switch {
		case r == 0:
			S.error(S.offset, "illegal character NUL")
		case r >= 0x80:
			// not ASCII
			r, w = utf8.DecodeRune(S.src[S.rdOffset:])
			if r == utf8.RuneError && w == 1 {
				S.error(S.offset, "illegal UTF-8 encoding")
			}
		}
		S.rdOffset += w
		S.ch = r
	} else {
		S.offset = len(S.src)
		if S.ch == '\n' {
			S.lineOffset = S.offset
			S.file.AddLine(S.offset)
		}
		S.ch = -1 // eof
	}
}

// The mode parameter to the Init function is a set of flags (or 0).
// They control scanner behavior.
//
const (
	ScanComments    = 1 << iota // return comments as COMMENT tokens
	dontInsertSemis             // do not automatically insert semicolons - for testing only
)

// Init prepares the scanner S to tokenize the text src by setting the
// scanner at the beginning of src. The scanner uses the file set file
// for position information and it adds line information for each line.
// It is ok to re-use the same file when re-scanning the same file as
// line information which is already present is ignored. Init causes a
// panic if the file size does not match the src size.
//
// Calls to Scan will use the error handler err if they encounter a
// syntax error and err is not nil. Also, for each error encountered,
// the Scanner field ErrorCount is incremented by one. The mode parameter
// determines how comments are handled.
//
// Note that Init may call err if there is an error in the first character
// of the file.
//
func (S *Scanner) Init(file *token.File, src []byte, err ErrorHandler, mode uint) {
	// Explicitly initialize all fields since a scanner may be reused.
	if file.Size() != len(src) {
		panic("file size does not match src len")
	}
	S.file = file
	S.dir, _ = filepath.Split(file.Name())
	S.src = src
	S.err = err
	S.mode = mode

	S.ch = ' '
	S.offset = 0
	S.rdOffset = 0
	S.lineOffset = 0
	S.insertSemi = false
	S.ErrorCount = 0

	S.next()
}

func (S *Scanner) error(offs int, msg string) {
	if S.err != nil {
		S.err.Error(S.file.Position(S.file.Pos(offs)), msg)
	}
	S.ErrorCount++
}

var prefix = []byte("//line ")

func (S *Scanner) interpretLineComment(text []byte) {
	if bytes.HasPrefix(text, prefix) {
		// get filename and line number, if any
		if i := bytes.LastIndex(text, []byte{':'}); i > 0 {
			if line, err := strconv.Atoi(string(text[i+1:])); err == nil && line > 0 {
				// valid //line filename:line comment;
				filename := filepath.Clean(string(text[len(prefix):i]))
				if !filepath.IsAbs(filename) {
					// make filename relative to current directory
					filename = filepath.Join(S.dir, filename)
				}
				// update scanner position
				S.file.AddLineInfo(S.lineOffset+len(text)+1, filename, line) // +len(text)+1 since comment applies to next line
			}
		}
	}
}

func (S *Scanner) scanComment() string {
	// initial '/' already consumed; S.ch == '/' || S.ch == '*'
	offs := S.offset - 1 // position of initial '/'

	if S.ch == '/' {
		//-style comment
		S.next()
		for S.ch != '\n' && S.ch >= 0 {
			S.next()
		}
		if offs == S.lineOffset {
			// comment starts at the beginning of the current line
			S.interpretLineComment(S.src[offs:S.offset])
		}
		goto exit
	}

	/*-style comment */
	S.next()
	for S.ch >= 0 {
		ch := S.ch
		S.next()
		if ch == '*' && S.ch == '/' {
			S.next()
			goto exit
		}
	}

	S.error(offs, "comment not terminated")

exit:
	return string(S.src[offs:S.offset])
}

func (S *Scanner) findLineEnd() bool {
	// initial '/' already consumed

	defer func(offs int) {
		// reset scanner state to where it was upon calling findLineEnd
		S.ch = '/'
		S.offset = offs
		S.rdOffset = offs + 1
		S.next() // consume initial '/' again
	}(S.offset - 1)

	// read ahead until a newline, EOF, or non-comment token is found
	for S.ch == '/' || S.ch == '*' {
		if S.ch == '/' {
			//-style comment always contains a newline
			return true
		}
		/*-style comment: look for newline */
		S.next()
		for S.ch >= 0 {
			ch := S.ch
			if ch == '\n' {
				return true
			}
			S.next()
			if ch == '*' && S.ch == '/' {
				S.next()
				break
			}
		}
		S.skipWhitespace() // S.insertSemi is set
		if S.ch < 0 || S.ch == '\n' {
			return true
		}
		if S.ch != '/' {
			// non-comment token
			return false
		}
		S.next() // consume '/'
	}

	return false
}

func isLetter(ch rune) bool {
	return 'a' <= ch && ch <= 'z' || 'A' <= ch && ch <= 'Z' || ch == '_' || ch >= 0x80 && unicode.IsLetter(ch)
}

func isDigit(ch rune) bool {
	return '0' <= ch && ch <= '9' || ch >= 0x80 && unicode.IsDigit(ch)
}

func (S *Scanner) scanIdentifier() string {
	offs := S.offset
	for isLetter(S.ch) || isDigit(S.ch) {
		S.next()
	}
	return string(S.src[offs:S.offset])
}

func digitVal(ch rune) int {
	switch {
	case '0' <= ch && ch <= '9':
		return int(ch - '0')
	case 'a' <= ch && ch <= 'f':
		return int(ch - 'a' + 10)
	case 'A' <= ch && ch <= 'F':
		return int(ch - 'A' + 10)
	}
	return 16 // larger than any legal digit val
}

func (S *Scanner) scanMantissa(base int) {
	for digitVal(S.ch) < base {
		S.next()
	}
}

func (S *Scanner) scanNumber(seenDecimalPoint bool) (token.Token, string) {
	// digitVal(S.ch) < 10
	offs := S.offset
	tok := token.INT

	if seenDecimalPoint {
		offs--
		tok = token.FLOAT
		S.scanMantissa(10)
		goto exponent
	}

	if S.ch == '0' {
		// int or float
		offs := S.offset
		S.next()
		if S.ch == 'x' || S.ch == 'X' {
			// hexadecimal int
			S.next()
			S.scanMantissa(16)
			if S.offset-offs <= 2 {
				// only scanned "0x" or "0X"
				S.error(offs, "illegal hexadecimal number")
			}
		} else {
			// octal int or float
			seenDecimalDigit := false
			S.scanMantissa(8)
			if S.ch == '8' || S.ch == '9' {
				// illegal octal int or float
				seenDecimalDigit = true
				S.scanMantissa(10)
			}
			if S.ch == '.' || S.ch == 'e' || S.ch == 'E' || S.ch == 'i' {
				goto fraction
			}
			// octal int
			if seenDecimalDigit {
				S.error(offs, "illegal octal number")
			}
		}
		goto exit
	}

	// decimal int or float
	S.scanMantissa(10)

fraction:
	if S.ch == '.' {
		tok = token.FLOAT
		S.next()
		S.scanMantissa(10)
	}

exponent:
	if S.ch == 'e' || S.ch == 'E' {
		tok = token.FLOAT
		S.next()
		if S.ch == '-' || S.ch == '+' {
			S.next()
		}
		S.scanMantissa(10)
	}

	if S.ch == 'i' {
		tok = token.IMAG
		S.next()
	}

exit:
	return tok, string(S.src[offs:S.offset])
}

func (S *Scanner) scanEscape(quote rune) {
	offs := S.offset

	var i, base, max uint32
	switch S.ch {
	case 'a', 'b', 'f', 'n', 'r', 't', 'v', '\\', quote:
		S.next()
		return
	case '0', '1', '2', '3', '4', '5', '6', '7':
		i, base, max = 3, 8, 255
	case 'x':
		S.next()
		i, base, max = 2, 16, 255
	case 'u':
		S.next()
		i, base, max = 4, 16, unicode.MaxRune
	case 'U':
		S.next()
		i, base, max = 8, 16, unicode.MaxRune
	default:
		S.next() // always make progress
		S.error(offs, "unknown escape sequence")
		return
	}

	var x uint32
	for ; i > 0 && S.ch != quote && S.ch >= 0; i-- {
		d := uint32(digitVal(S.ch))
		if d >= base {
			S.error(S.offset, "illegal character in escape sequence")
			break
		}
		x = x*base + d
		S.next()
	}
	// in case of an error, consume remaining chars
	for ; i > 0 && S.ch != quote && S.ch >= 0; i-- {
		S.next()
	}
	if x > max || 0xd800 <= x && x < 0xe000 {
		S.error(offs, "escape sequence is invalid Unicode code point")
	}
}

func (S *Scanner) scanChar() string {
	// '\'' opening already consumed
	offs := S.offset - 1

	n := 0
	for S.ch != '\'' {
		ch := S.ch
		n++
		S.next()
		if ch == '\n' || ch < 0 {
			S.error(offs, "character literal not terminated")
			n = 1
			break
		}
		if ch == '\\' {
			S.scanEscape('\'')
		}
	}

	S.next()

	if n != 1 {
		S.error(offs, "illegal character literal")
	}

	return string(S.src[offs:S.offset])
}

func (S *Scanner) scanString() string {
	// '"' opening already consumed
	offs := S.offset - 1

	for S.ch != '"' {
		ch := S.ch
		S.next()
		if ch == '\n' || ch < 0 {
			S.error(offs, "string not terminated")
			break
		}
		if ch == '\\' {
			S.scanEscape('"')
		}
	}

	S.next()

	return string(S.src[offs:S.offset])
}

func stripCR(b []byte) []byte {
	c := make([]byte, len(b))
	i := 0
	for _, ch := range b {
		if ch != '\r' {
			c[i] = ch
			i++
		}
	}
	return c[:i]
}

func (S *Scanner) scanRawString() string {
	// '`' opening already consumed
	offs := S.offset - 1

	hasCR := false
	for S.ch != '`' {
		ch := S.ch
		S.next()
		if ch == '\r' {
			hasCR = true
		}
		if ch < 0 {
			S.error(offs, "string not terminated")
			break
		}
	}

	S.next()

	lit := S.src[offs:S.offset]
	if hasCR {
		lit = stripCR(lit)
	}

	return string(lit)
}

func (S *Scanner) skipWhitespace() {
	for S.ch == ' ' || S.ch == '\t' || S.ch == '\n' && !S.insertSemi || S.ch == '\r' {
		S.next()
	}
}

// Helper functions for scanning multi-byte tokens such as >> += >>= .
// Different routines recognize different length tok_i based on matches
// of ch_i. If a token ends in '=', the result is tok1 or tok3
// respectively. Otherwise, the result is tok0 if there was no other
// matching character, or tok2 if the matching character was ch2.

func (S *Scanner) switch2(tok0, tok1 token.Token) token.Token {
	if S.ch == '=' {
		S.next()
		return tok1
	}
	return tok0
}

func (S *Scanner) switch3(tok0, tok1 token.Token, ch2 rune, tok2 token.Token) token.Token {
	if S.ch == '=' {
		S.next()
		return tok1
	}
	if S.ch == ch2 {
		S.next()
		return tok2
	}
	return tok0
}

func (S *Scanner) switch4(tok0, tok1 token.Token, ch2 rune, tok2, tok3 token.Token) token.Token {
	if S.ch == '=' {
		S.next()
		return tok1
	}
	if S.ch == ch2 {
		S.next()
		if S.ch == '=' {
			S.next()
			return tok3
		}
		return tok2
	}
	return tok0
}

// Scan scans the next token and returns the token position, the token,
// and its literal string if applicable. The source end is indicated by
// token.EOF.
//
// If the returned token is a literal (token.IDENT, token.INT, token.FLOAT,
// token.IMAG, token.CHAR, token.STRING) or token.COMMENT, the literal string
// has the corresponding value.
//
// If the returned token is token.SEMICOLON, the corresponding
// literal string is ";" if the semicolon was present in the source,
// and "\n" if the semicolon was inserted because of a newline or
// at EOF.
//
// If the returned token is token.ILLEGAL, the literal string is the
// offending character.
//
// In all other cases, Scan returns an empty literal string.
//
// For more tolerant parsing, Scan will return a valid token if
// possible even if a syntax error was encountered. Thus, even
// if the resulting token sequence contains no illegal tokens,
// a client may not assume that no error occurred. Instead it
// must check the scanner's ErrorCount or the number of calls
// of the error handler, if there was one installed.
//
// Scan adds line information to the file added to the file
// set with Init. Token positions are relative to that file
// and thus relative to the file set.
//
func (S *Scanner) Scan() (pos token.Pos, tok token.Token, lit string) {
scanAgain:
	S.skipWhitespace()

	// current token start
	pos = S.file.Pos(S.offset)

	// determine token value
	insertSemi := false
	switch ch := S.ch; {
	case isLetter(ch):
		lit = S.scanIdentifier()
		tok = token.Lookup(lit)
		switch tok {
		case token.IDENT, token.BREAK, token.CONTINUE, token.FALLTHROUGH, token.RETURN:
			insertSemi = true
		}
	case digitVal(ch) < 10:
		insertSemi = true
		tok, lit = S.scanNumber(false)
	default:
		S.next() // always make progress
		switch ch {
		case -1:
			if S.insertSemi {
				S.insertSemi = false // EOF consumed
				return pos, token.SEMICOLON, "\n"
			}
			tok = token.EOF
		case '\n':
			// we only reach here if S.insertSemi was
			// set in the first place and exited early
			// from S.skipWhitespace()
			S.insertSemi = false // newline consumed
			return pos, token.SEMICOLON, "\n"
		case '"':
			insertSemi = true
			tok = token.STRING
			lit = S.scanString()
		case '\'':
			insertSemi = true
			tok = token.CHAR
			lit = S.scanChar()
		case '`':
			insertSemi = true
			tok = token.STRING
			lit = S.scanRawString()
		case ':':
			tok = S.switch2(token.COLON, token.DEFINE)
		case '.':
			if digitVal(S.ch) < 10 {
				insertSemi = true
				tok, lit = S.scanNumber(true)
			} else if S.ch == '.' {
				S.next()
				if S.ch == '.' {
					S.next()
					tok = token.ELLIPSIS
				}
			} else {
				tok = token.PERIOD
			}
		case ',':
			tok = token.COMMA
		case ';':
			tok = token.SEMICOLON
			lit = ";"
		case '(':
			tok = token.LPAREN
		case ')':
			insertSemi = true
			tok = token.RPAREN
		case '[':
			tok = token.LBRACK
		case ']':
			insertSemi = true
			tok = token.RBRACK
		case '{':
			tok = token.LBRACE
		case '}':
			insertSemi = true
			tok = token.RBRACE
		case '+':
			tok = S.switch3(token.ADD, token.ADD_ASSIGN, '+', token.INC)
			if tok == token.INC {
				insertSemi = true
			}
		case '-':
			tok = S.switch3(token.SUB, token.SUB_ASSIGN, '-', token.DEC)
			if tok == token.DEC {
				insertSemi = true
			}
		case '*':
			tok = S.switch2(token.MUL, token.MUL_ASSIGN)
		case '/':
			if S.ch == '/' || S.ch == '*' {
				// comment
				if S.insertSemi && S.findLineEnd() {
					// reset position to the beginning of the comment
					S.ch = '/'
					S.offset = S.file.Offset(pos)
					S.rdOffset = S.offset + 1
					S.insertSemi = false // newline consumed
					return pos, token.SEMICOLON, "\n"
				}
				lit = S.scanComment()
				if S.mode&ScanComments == 0 {
					// skip comment
					S.insertSemi = false // newline consumed
					goto scanAgain
				}
				tok = token.COMMENT
			} else {
				tok = S.switch2(token.QUO, token.QUO_ASSIGN)
			}
		case '%':
			tok = S.switch2(token.REM, token.REM_ASSIGN)
		case '^':
			tok = S.switch2(token.XOR, token.XOR_ASSIGN)
		case '<':
			if S.ch == '-' {
				S.next()
				tok = token.ARROW
			} else {
				tok = S.switch4(token.LSS, token.LEQ, '<', token.SHL, token.SHL_ASSIGN)
			}
		case '>':
			tok = S.switch4(token.GTR, token.GEQ, '>', token.SHR, token.SHR_ASSIGN)
		case '=':
			tok = S.switch2(token.ASSIGN, token.EQL)
		case '!':
			tok = S.switch2(token.NOT, token.NEQ)
		case '&':
			if S.ch == '^' {
				S.next()
				tok = S.switch2(token.AND_NOT, token.AND_NOT_ASSIGN)
			} else {
				tok = S.switch3(token.AND, token.AND_ASSIGN, '&', token.LAND)
			}
		case '|':
			tok = S.switch3(token.OR, token.OR_ASSIGN, '|', token.LOR)
		default:
			S.error(S.file.Offset(pos), fmt.Sprintf("illegal character %#U", ch))
			insertSemi = S.insertSemi // preserve insertSemi info
			tok = token.ILLEGAL
			lit = string(ch)
		}
	}
	if S.mode&dontInsertSemis == 0 {
		S.insertSemi = insertSemi
	}

	return
}
