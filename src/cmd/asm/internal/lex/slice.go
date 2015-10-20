// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lex

import "text/scanner"

// A Slice reads from a slice of Tokens.
type Slice struct {
	tokens   []Token
	fileName string
	line     int
	pos      int
}

func NewSlice(fileName string, line int, tokens []Token) *Slice {
	return &Slice{
		tokens:   tokens,
		fileName: fileName,
		line:     line,
		pos:      -1, // Next will advance to zero.
	}
}

func (s *Slice) Next() ScanToken {
	s.pos++
	if s.pos >= len(s.tokens) {
		return scanner.EOF
	}
	return s.tokens[s.pos].ScanToken
}

func (s *Slice) Text() string {
	return s.tokens[s.pos].text
}

func (s *Slice) File() string {
	return s.fileName
}

func (s *Slice) Line() int {
	return s.line
}

func (s *Slice) Col() int {
	// TODO: Col is only called when defining a macro and all it cares about is increasing
	// position to discover whether there is a blank before the parenthesis.
	// We only get here if defining a macro inside a macro.
	// This imperfect implementation means we cannot tell the difference between
	//	#define A #define B(x) x
	// and
	//	#define A #define B (x) x
	// The first has definition of B has an argument, the second doesn't. Because we let
	// text/scanner strip the blanks for us, this is extremely rare, hard to fix, and not worth it.
	return s.pos
}

func (s *Slice) SetPos(line int, file string) {
	// Cannot happen because we only have slices of already-scanned
	// text, but be prepared.
	s.line = line
	s.fileName = file
}

func (s *Slice) Close() {
}
