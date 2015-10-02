// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lex

import (
	"io"
	"os"
	"strings"
	"text/scanner"
	"unicode"
)

// A Tokenizer is a simple wrapping of text/scanner.Scanner, configured
// for our purposes and made a TokenReader. It forms the lowest level,
// turning text from readers into tokens.
type Tokenizer struct {
	tok      ScanToken
	s        *scanner.Scanner
	line     int
	fileName string
	file     *os.File // If non-nil, file descriptor to close.
}

func NewTokenizer(name string, r io.Reader, file *os.File) *Tokenizer {
	var s scanner.Scanner
	s.Init(r)
	// Newline is like a semicolon; other space characters are fine.
	s.Whitespace = 1<<'\t' | 1<<'\r' | 1<<' '
	// Don't skip comments: we need to count newlines.
	s.Mode = scanner.ScanChars |
		scanner.ScanFloats |
		scanner.ScanIdents |
		scanner.ScanInts |
		scanner.ScanStrings |
		scanner.ScanComments
	s.Position.Filename = name
	s.IsIdentRune = isIdentRune
	if file != nil {
		linkCtxt.LineHist.Push(histLine, name)
	}
	return &Tokenizer{
		s:        &s,
		line:     1,
		fileName: name,
		file:     file,
	}
}

// We want center dot (·) and division slash (∕) to work as identifier characters.
func isIdentRune(ch rune, i int) bool {
	if unicode.IsLetter(ch) {
		return true
	}
	switch ch {
	case '_': // Underscore; traditional.
		return true
	case '\u00B7': // Represents the period in runtime.exit. U+00B7 '·' middle dot
		return true
	case '\u2215': // Represents the slash in runtime/debug.setGCPercent. U+2215 '∕' division slash
		return true
	}
	// Digits are OK only after the first character.
	return i > 0 && unicode.IsDigit(ch)
}

func (t *Tokenizer) Text() string {
	switch t.tok {
	case LSH:
		return "<<"
	case RSH:
		return ">>"
	case ARR:
		return "->"
	case ROT:
		return "@>"
	}
	return t.s.TokenText()
}

func (t *Tokenizer) File() string {
	return t.fileName
}

func (t *Tokenizer) Line() int {
	return t.line
}

func (t *Tokenizer) Col() int {
	return t.s.Pos().Column
}

func (t *Tokenizer) SetPos(line int, file string) {
	t.line = line
	t.fileName = file
}

func (t *Tokenizer) Next() ScanToken {
	s := t.s
	for {
		t.tok = ScanToken(s.Scan())
		if t.tok != scanner.Comment {
			break
		}
		length := strings.Count(s.TokenText(), "\n")
		t.line += length
		histLine += length
		// TODO: If we ever have //go: comments in assembly, will need to keep them here.
		// For now, just discard all comments.
	}
	switch t.tok {
	case '\n':
		if t.file != nil {
			histLine++
		}
		t.line++
	case '-':
		if s.Peek() == '>' {
			s.Next()
			t.tok = ARR
			return ARR
		}
	case '@':
		if s.Peek() == '>' {
			s.Next()
			t.tok = ROT
			return ROT
		}
	case '<':
		if s.Peek() == '<' {
			s.Next()
			t.tok = LSH
			return LSH
		}
	case '>':
		if s.Peek() == '>' {
			s.Next()
			t.tok = RSH
			return RSH
		}
	}
	return t.tok
}

func (t *Tokenizer) Close() {
	if t.file != nil {
		t.file.Close()
		// It's an open file, so pop the line history.
		linkCtxt.LineHist.Pop(histLine)
	}
}
