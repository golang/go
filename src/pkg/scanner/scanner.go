// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// A scanner and tokenizer for UTF-8-encoded text.  Takes an io.Reader
// providing the source, which then can be tokenized through repeated calls
// to the Scan function.  For compatibility with existing tools, the NUL
// character is not allowed (implementation restriction).
//
// By default, a Scanner skips white space and Go comments and recognizes all
// literals as defined by the Go language specification.  It may be
// customized to recognize only a subset of those literals and to recognize
// different white space characters.
//
// Basic usage pattern:
//
//	var s scanner.Scanner
//	s.Init(src)
//	tok := s.Scan()
//	for tok != scanner.EOF {
//		// do something with tok
//		tok = s.Scan()
//	}
//
package scanner

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"unicode"
	"utf8"
)


// A source position is represented by a Position value.
// A position is valid if Line > 0.
type Position struct {
	Filename string // filename, if any
	Offset   int    // byte offset, starting at 0
	Line     int    // line number, starting at 1
	Column   int    // column number, starting at 0 (character count per line)
}


// IsValid returns true if the position is valid.
func (pos *Position) IsValid() bool { return pos.Line > 0 }


func (pos Position) String() string {
	s := pos.Filename
	if pos.IsValid() {
		if s != "" {
			s += ":"
		}
		s += fmt.Sprintf("%d:%d", pos.Line, pos.Column)
	}
	if s == "" {
		s = "???"
	}
	return s
}


// Predefined mode bits to control recognition of tokens. For instance,
// to configure a Scanner such that it only recognizes (Go) identifiers,
// integers, and skips comments, set the Scanner's Mode field to:
//
//	ScanIdents | ScanInts | SkipComments
//
const (
	ScanIdents     = 1 << -Ident
	ScanInts       = 1 << -Int
	ScanFloats     = 1 << -Float // includes Ints
	ScanChars      = 1 << -Char
	ScanStrings    = 1 << -String
	ScanRawStrings = 1 << -RawString
	ScanComments   = 1 << -Comment
	SkipComments   = 1 << -skipComment // if set with ScanComments, comments become white space
	GoTokens       = ScanIdents | ScanFloats | ScanChars | ScanStrings | ScanRawStrings | ScanComments | SkipComments
)


// The result of Scan is one of the following tokens or a Unicode character.
const (
	EOF = -(iota + 1)
	Ident
	Int
	Float
	Char
	String
	RawString
	Comment
	skipComment
)


var tokenString = map[int]string{
	EOF:       "EOF",
	Ident:     "Ident",
	Int:       "Int",
	Float:     "Float",
	Char:      "Char",
	String:    "String",
	RawString: "RawString",
	Comment:   "Comment",
}


// TokenString returns a (visible) string for a token or Unicode character.
func TokenString(tok int) string {
	if s, found := tokenString[tok]; found {
		return s
	}
	return fmt.Sprintf("U+%04X", tok)
}


// GoWhitespace is the default value for the Scanner's Whitespace field.
// Its value selects Go's white space characters.
const GoWhitespace = 1<<'\t' | 1<<'\n' | 1<<'\r' | 1<<' '


const bufLen = 1024 // at least utf8.UTFMax

// A Scanner implements reading of Unicode characters and tokens from an io.Reader.
type Scanner struct {
	// Input
	src io.Reader

	// Source buffer
	srcBuf [bufLen + 1]byte // +1 for sentinel for common case of s.next()
	srcPos int              // reading position (srcBuf index)
	srcEnd int              // source end (srcBuf index)

	// Source position
	srcBufOffset int // byte offset of srcBuf[0] in source
	line         int // newline count + 1
	column       int // character count on line

	// Token text buffer
	// Typically, token text is stored completely in srcBuf, but in general
	// the token text's head may be buffered in tokBuf while the token text's
	// tail is stored in srcBuf.
	tokBuf bytes.Buffer // token text head that is not in srcBuf anymore
	tokPos int          // token text tail position (srcBuf index)
	tokEnd int          // token text tail end (srcBuf index)

	// One character look-ahead
	ch int // character before current srcPos

	// Error is called for each error encountered. If no Error
	// function is set, the error is reported to os.Stderr.
	Error func(s *Scanner, msg string)

	// ErrorCount is incremented by one for each error encountered.
	ErrorCount int

	// The Mode field controls which tokens are recognized. For instance,
	// to recognize Ints, set the ScanInts bit in Mode. The field may be
	// changed at any time.
	Mode uint

	// The Whitespace field controls which characters are recognized
	// as white space. To recognize a character ch <= ' ' as white space,
	// set the ch'th bit in Whitespace (the Scanner's behavior is undefined
	// for values ch > ' '). The field may be changed at any time.
	Whitespace uint64

	// Current token position. The Offset, Line, and Column fields
	// are set by Scan(); the Filename field is left untouched by the
	// Scanner.
	Position
}


// Init initializes a Scanner with a new source and returns itself.
// Error is set to nil, ErrorCount is set to 0, Mode is set to GoTokens,
// and Whitespace is set to GoWhitespace.
func (s *Scanner) Init(src io.Reader) *Scanner {
	s.src = src

	// initialize source buffer
	s.srcBuf[0] = utf8.RuneSelf // sentinel
	s.srcPos = 0
	s.srcEnd = 0

	// initialize source position
	s.srcBufOffset = 0
	s.line = 1
	s.column = 0

	// initialize token text buffer
	s.tokPos = -1

	// initialize one character look-ahead
	s.ch = s.next()

	// initialize public fields
	s.Error = nil
	s.ErrorCount = 0
	s.Mode = GoTokens
	s.Whitespace = GoWhitespace

	return s
}


// next reads and returns the next Unicode character. It is designed such
// that only a minimal amount of work needs to be done in the common ASCII
// case (one test to check for both ASCII and end-of-buffer, and one test
// to check for newlines).
func (s *Scanner) next() int {
	ch := int(s.srcBuf[s.srcPos])

	if ch >= utf8.RuneSelf {
		// uncommon case: not ASCII or not enough bytes
		for s.srcPos+utf8.UTFMax > s.srcEnd && !utf8.FullRune(s.srcBuf[s.srcPos:s.srcEnd]) {
			// not enough bytes: read some more, but first
			// save away token text if any
			if s.tokPos >= 0 {
				s.tokBuf.Write(s.srcBuf[s.tokPos:s.srcPos])
				s.tokPos = 0
			}
			// move unread bytes to beginning of buffer
			copy(s.srcBuf[0:], s.srcBuf[s.srcPos:s.srcEnd])
			s.srcBufOffset += s.srcPos
			// read more bytes
			i := s.srcEnd - s.srcPos
			n, err := s.src.Read(s.srcBuf[i:bufLen])
			s.srcEnd = i + n
			s.srcPos = 0
			s.srcBuf[s.srcEnd] = utf8.RuneSelf // sentinel
			if err != nil {
				if s.srcEnd == 0 {
					return EOF
				}
				if err != os.EOF {
					s.error(err.String())
					break
				}
			}
		}
		// at least one byte
		ch = int(s.srcBuf[s.srcPos])
		if ch >= utf8.RuneSelf {
			// uncommon case: not ASCII
			var width int
			ch, width = utf8.DecodeRune(s.srcBuf[s.srcPos:s.srcEnd])
			if ch == utf8.RuneError && width == 1 {
				s.error("illegal UTF-8 encoding")
			}
			s.srcPos += width - 1
		}
	}

	s.srcPos++
	s.column++
	switch ch {
	case 0:
		// implementation restriction for compatibility with other tools
		s.error("illegal character NUL")
	case '\n':
		s.line++
		s.column = 0
	}

	return ch
}


// Next reads and returns the next Unicode character.
// It returns EOF at the end of the source. It reports
// a read error by calling s.Error, if set, or else
// prints an error message to os.Stderr. Next does not
// update the Scanner's Position field; use Pos() to
// get the current position.
func (s *Scanner) Next() int {
	s.tokPos = -1 // don't collect token text
	ch := s.ch
	s.ch = s.next()
	return ch
}


// Peek returns the next Unicode character in the source without advancing
// the scanner. It returns EOF if the scanner's position is at the last
// character of the source.
func (s *Scanner) Peek() int {
	return s.ch
}


func (s *Scanner) error(msg string) {
	s.ErrorCount++
	if s.Error != nil {
		s.Error(s, msg)
		return
	}
	fmt.Fprintf(os.Stderr, "%s: %s", s.Position, msg)
}


func (s *Scanner) scanIdentifier() int {
	ch := s.next() // read character after first '_' or letter
	for ch == '_' || unicode.IsLetter(ch) || unicode.IsDigit(ch) {
		ch = s.next()
	}
	return ch
}


func digitVal(ch int) int {
	switch {
	case '0' <= ch && ch <= '9':
		return ch - '0'
	case 'a' <= ch && ch <= 'f':
		return ch - 'a' + 10
	case 'A' <= ch && ch <= 'F':
		return ch - 'A' + 10
	}
	return 16 // larger than any legal digit val
}


func isDecimal(ch int) bool { return '0' <= ch && ch <= '9' }


func (s *Scanner) scanMantissa(ch int) int {
	for isDecimal(ch) {
		ch = s.next()
	}
	return ch
}


func (s *Scanner) scanFraction(ch int) int {
	if ch == '.' {
		ch = s.scanMantissa(s.next())
	}
	return ch
}


func (s *Scanner) scanExponent(ch int) int {
	if ch == 'e' || ch == 'E' {
		ch = s.next()
		if ch == '-' || ch == '+' {
			ch = s.next()
		}
		ch = s.scanMantissa(ch)
	}
	return ch
}


func (s *Scanner) scanNumber(ch int) (int, int) {
	// isDecimal(ch)
	if ch == '0' {
		// int or float
		ch = s.next()
		if ch == 'x' || ch == 'X' {
			// hexadecimal int
			ch = s.next()
			for digitVal(ch) < 16 {
				ch = s.next()
			}
		} else {
			// octal int or float
			seenDecimalDigit := false
			for isDecimal(ch) {
				if ch > '7' {
					seenDecimalDigit = true
				}
				ch = s.next()
			}
			if s.Mode&ScanFloats != 0 && (ch == '.' || ch == 'e' || ch == 'E') {
				// float
				ch = s.scanFraction(ch)
				ch = s.scanExponent(ch)
				return Float, ch
			}
			// octal int
			if seenDecimalDigit {
				s.error("illegal octal number")
			}
		}
		return Int, ch
	}
	// decimal int or float
	ch = s.scanMantissa(ch)
	if s.Mode&ScanFloats != 0 && (ch == '.' || ch == 'e' || ch == 'E') {
		// float
		ch = s.scanFraction(ch)
		ch = s.scanExponent(ch)
		return Float, ch
	}
	return Int, ch
}


func (s *Scanner) scanDigits(ch, base, n int) int {
	for n > 0 && digitVal(ch) < base {
		ch = s.next()
		n--
	}
	if n > 0 {
		s.error("illegal char escape")
	}
	return ch
}


func (s *Scanner) scanEscape(quote int) int {
	ch := s.next() // read character after '/'
	switch ch {
	case 'a', 'b', 'f', 'n', 'r', 't', 'v', '\\', quote:
		// nothing to do
		ch = s.next()
	case '0', '1', '2', '3', '4', '5', '6', '7':
		ch = s.scanDigits(ch, 8, 3)
	case 'x':
		ch = s.scanDigits(s.next(), 16, 2)
	case 'u':
		ch = s.scanDigits(s.next(), 16, 4)
	case 'U':
		ch = s.scanDigits(s.next(), 16, 8)
	default:
		s.error("illegal char escape")
	}
	return ch
}


func (s *Scanner) scanString(quote int) (n int) {
	ch := s.next() // read character after quote
	for ch != quote {
		if ch == '\n' || ch < 0 {
			s.error("literal not terminated")
			return
		}
		if ch == '\\' {
			ch = s.scanEscape(quote)
		} else {
			ch = s.next()
		}
		n++
	}
	return
}


func (s *Scanner) scanRawString() {
	ch := s.next() // read character after '`'
	for ch != '`' {
		if ch < 0 {
			s.error("literal not terminated")
			return
		}
		ch = s.next()
	}
}


func (s *Scanner) scanChar() {
	if s.scanString('\'') != 1 {
		s.error("illegal char literal")
	}
}


func (s *Scanner) scanLineComment() {
	ch := s.next() // read character after "//"
	for ch != '\n' {
		if ch < 0 {
			s.error("comment not terminated")
			return
		}
		ch = s.next()
	}
}


func (s *Scanner) scanGeneralComment() {
	ch := s.next() // read character after "/*"
	for {
		if ch < 0 {
			s.error("comment not terminated")
			return
		}
		ch0 := ch
		ch = s.next()
		if ch0 == '*' && ch == '/' {
			break
		}
	}
}


func (s *Scanner) scanComment(ch int) {
	// ch == '/' || ch == '*'
	if ch == '/' {
		s.scanLineComment()
		return
	}
	s.scanGeneralComment()
}


// Scan reads the next token or Unicode character from source and returns it.
// It only recognizes tokens t for which the respective Mode bit (1<<-t) is set.
// It returns EOF at the end of the source. It reports scanner errors (read and
// token errors) by calling s.Error, if set; otherwise it prints an error message
// to os.Stderr.
func (s *Scanner) Scan() int {
	ch := s.ch

	// reset token text position
	s.tokPos = -1

redo:
	// skip white space
	for s.Whitespace&(1<<uint(ch)) != 0 {
		ch = s.next()
	}

	// start collecting token text
	s.tokBuf.Reset()
	s.tokPos = s.srcPos - 1

	// set token position
	s.Offset = s.srcBufOffset + s.tokPos
	s.Line = s.line
	s.Column = s.column

	// determine token value
	tok := ch
	switch {
	case unicode.IsLetter(ch) || ch == '_':
		if s.Mode&ScanIdents != 0 {
			tok = Ident
			ch = s.scanIdentifier()
		} else {
			ch = s.next()
		}
	case isDecimal(ch):
		if s.Mode&(ScanInts|ScanFloats) != 0 {
			tok, ch = s.scanNumber(ch)
		} else {
			ch = s.next()
		}
	default:
		switch ch {
		case '"':
			if s.Mode&ScanStrings != 0 {
				s.scanString('"')
				tok = String
			}
			ch = s.next()
		case '\'':
			if s.Mode&ScanChars != 0 {
				s.scanChar()
				tok = Char
			}
			ch = s.next()
		case '.':
			ch = s.next()
			if isDecimal(ch) && s.Mode&ScanFloats != 0 {
				tok = Float
				ch = s.scanMantissa(ch)
				ch = s.scanExponent(ch)
			}
		case '/':
			ch = s.next()
			if (ch == '/' || ch == '*') && s.Mode&ScanComments != 0 {
				if s.Mode&SkipComments != 0 {
					s.tokPos = -1 // don't collect token text
					s.scanComment(ch)
					ch = s.next()
					goto redo
				}
				s.scanComment(ch)
				tok = Comment
				ch = s.next()
			}
		case '`':
			if s.Mode&ScanRawStrings != 0 {
				s.scanRawString()
				tok = String
			}
			ch = s.next()
		default:
			ch = s.next()
		}
	}

	// end of token text
	s.tokEnd = s.srcPos - 1

	s.ch = ch
	return tok
}


// Position returns the current source position. If called before Next()
// or Scan(), it returns the position of the next Unicode character or token
// returned by these functions. If called afterwards, it returns the position
// immediately after the last character of the most recent token or character
// scanned.
func (s *Scanner) Pos() Position {
	return Position{
		s.Filename,
		s.srcBufOffset + s.srcPos - 1,
		s.line,
		s.column,
	}
}


// TokenText returns the string corresponding to the most recently scanned token.
// Valid after calling Scan().
func (s *Scanner) TokenText() string {
	if s.tokPos < 0 {
		// no token text
		return ""
	}

	if s.tokEnd < 0 {
		// if EOF was reached, s.tokEnd is set to -1 (s.srcPos == 0)
		s.tokEnd = s.tokPos
	}

	if s.tokBuf.Len() == 0 {
		// common case: the entire token text is still in srcBuf
		return string(s.srcBuf[s.tokPos:s.tokEnd])
	}

	// part of the token text was saved in tokBuf: save the rest in
	// tokBuf as well and return its content
	s.tokBuf.Write(s.srcBuf[s.tokPos:s.tokEnd])
	s.tokPos = s.tokEnd // ensure idempotency of TokenText() call
	return s.tokBuf.String()
}
