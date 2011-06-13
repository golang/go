// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package template

import (
	"fmt"
	"strings"
	"unicode"
	"utf8"
)

// item represents a token or text string returned from the scanner.
type item struct {
	typ itemType
	val string
}

// itemType identifies the type of lex item.
type itemType int

const (
	itemError      itemType = iota // error occurred; value is text of error
	itemText                       // plain text
	itemLeftMeta                   // left meta-string
	itemRightMeta                  // right meta-string
	itemPipe                       // pipe symbol
	itemIdentifier                 // alphanumeric identifier
	itemNumber                     // number
	itemRawString                  // raw quoted string (includes quotes)
	itemString                     // quoted string (includes quotes)
	itemEOF
)

const eof = -1

// stateFn represents the state of the scanner as a function that returns the next state.
type stateFn func(*lexer) stateFn

// lexer holds the state of the scanner.
type lexer struct {
	name  string    // the name of the input; used only for error reports.
	input string    // the string being scanned.
	pos   int       // current position in the input.
	start int       // start position of this item.
	width int       // width of last rune read from input.
	items chan item // channel of scanned items.
}

// next returns the next rune in the input.
func (l *lexer) next() (rune int) {
	if l.pos >= len(l.input) {
		return eof
	}
	rune, l.width = utf8.DecodeRuneInString(l.input[l.pos:])
	l.pos += l.width
	return rune
}

// peek returns but does not consume the next rune in the input.
func (l *lexer) peek() int {
	rune := l.next()
	l.backup()
	return rune
}

// backup steps back one rune. Can only be called once per call of next.
func (l *lexer) backup() {
	l.pos -= l.width
}

// emit passes an item back to the client.
func (l *lexer) emit(t itemType) {
	start := l.start
	l.start = l.pos
	l.items <- item{t, l.input[start:l.pos]}
}

// ignore discards whatever input is before this point.
func (l *lexer) ignore() {
	l.start = l.pos
}

// accept consumes the next rune if it's from the valid set.
func (l *lexer) accept(valid string) bool {
	if strings.IndexRune(valid, l.next()) >= 0 {
		return true
	}
	l.backup()
	return false
}

// acceptRun consumes a run of runes from the valid set.
func (l *lexer) acceptRun(valid string) {
	for strings.IndexRune(valid, l.next()) >= 0 {
	}
	l.backup()
}

// lineNumber reports which line we're on. Doing it this way
// means we don't have to worry about peek double counting.
func (l *lexer) lineNumber() int {
	return 1 + strings.Count(l.input[:l.pos], "\n")
}

// error returns an error token and terminates the scan by passing
// back a nil pointer that will be the next state, terminating l.run.
func (l *lexer) error(format string, args ...interface{}) stateFn {
	format = fmt.Sprintf("%s:%d %s", l.name, l.lineNumber(), format)
	l.items <- item{itemError, fmt.Sprintf(format, args...)}
	return nil
}

// run lexes the input by execute state functions until nil.
func (l *lexer) run() {
	for state := lexText; state != nil; {
		state = state(l)
	}
	close(l.items)
}

// lex launches a new scanner and returns the channel of items.
func lex(name, input string) chan item {
	l := &lexer{
		name:  name,
		input: input,
		items: make(chan item),
	}
	go l.run()
	return l.items
}

// state functions

const leftMeta = "{{"
const rightMeta = "}}"

// lexText scans until a metacharacter
func lexText(l *lexer) stateFn {
	for {
		if strings.HasPrefix(l.input[l.pos:], leftMeta) {
			if l.pos > l.start {
				l.emit(itemText)
			}
			return lexLeftMeta
		}
		if l.next() == eof {
			break
		}
	}
	// Correctly reached EOF.
	if l.pos > l.start {
		l.emit(itemText)
	}
	l.emit(itemEOF)
	return nil
}

// leftMeta scans the left "metacharacter", which is known to be present.
func lexLeftMeta(l *lexer) stateFn {
	l.pos += len(leftMeta)
	l.emit(itemLeftMeta)
	return lexInsideAction
}

// rightMeta scans the right "metacharacter", which is known to be present.
func lexRightMeta(l *lexer) stateFn {
	l.pos += len(rightMeta)
	l.emit(itemRightMeta)
	return lexText
}

// lexInsideAction scans the elements inside "metacharacters".
func lexInsideAction(l *lexer) stateFn {
	// Either number, quoted string, or identifier.
	// Spaces separate and are ignored.
	// Pipe symbols separate and are emitted.
	for {
		if strings.HasPrefix(l.input[l.pos:], rightMeta) {
			return lexRightMeta
		}
		switch r := l.next(); {
		case r == eof || r == '\n':
			return l.error("unclosed action")
		case isSpace(r):
			l.ignore()
		case r == '|':
			l.emit(itemPipe)
		case r == '"':
			return lexQuote
		case r == '`':
			return lexRawQuote
		case r == '+' || r == '-' || r == '.' || ('0' <= r && r <= '9'):
			l.backup()
			return lexNumber
		case isAlphaNumeric(r):
			l.backup()
			return lexIdentifier
		default:
			return l.error("unrecognized character in action: %#U", r)
		}
	}
	return nil
}

// lexIdentifier scans an alphanumeric.
func lexIdentifier(l *lexer) stateFn {
Loop:
	for {
		switch r := l.next(); {
		case isAlphaNumeric(r):
			// absorb
		default:
			l.backup()
			l.emit(itemIdentifier)
			break Loop
		}
	}
	return lexInsideAction
}

// lexNumber scans a number: decimal, octal, hex, float, or imaginary.  This
// isn't a perfect number scanner - for instance it accepts "." and "0x0.2"
// and "089" - but when it's wrong the input is invalid and the parser (via
// strconv) will notice.
// TODO: without expressions you can do imaginary but not complex.
func lexNumber(l *lexer) stateFn {
	// Optional leading sign.
	l.accept("+-")
	// Is it hex?
	digits := "0123456789"
	if l.accept("0") && l.accept("xX") {
		digits = "0123456789abcdefABCDEF"
	}
	l.acceptRun(digits)
	if l.accept(".") {
		l.acceptRun(digits)
	}
	if l.accept("eE") {
		l.accept("+-")
		l.acceptRun("0123456789")
	}
	// Is it imaginary?
	l.accept("i")
	// Next thing mustn't be alphanumeric.
	if isAlphaNumeric(l.peek()) {
		l.next()
		return l.error("bad number syntax: %q", l.input[l.start:l.pos])
	}
	l.emit(itemNumber)
	return lexInsideAction
}

// lexQuote scans a quoted string.
func lexQuote(l *lexer) stateFn {
Loop:
	for {
		switch l.next() {
		case '\\':
			if r := l.next(); r != eof && r != '\n' {
				break
			}
			fallthrough
		case eof, '\n':
			return l.error("unterminated quoted string")
		case '"':
			break Loop
		}
	}
	l.emit(itemString)
	return lexInsideAction
}

// lexRawQuote scans a raw quoted string.
func lexRawQuote(l *lexer) stateFn {
Loop:
	for {
		switch l.next() {
		case eof, '\n':
			return l.error("unterminated raw quoted string")
		case '`':
			break Loop
		}
	}
	l.emit(itemRawString)
	return lexInsideAction
}

// isSpace reports whether r is a space character.
func isSpace(r int) bool {
	switch r {
	case ' ', '\t', '\n', '\r':
		return true
	}
	return false
}

// isAlphaNumeric reports whether r is an alphabetic, digit, or underscore.
func isAlphaNumeric(r int) bool {
	return r == '_' || unicode.IsLetter(r) || unicode.IsDigit(r)
}
