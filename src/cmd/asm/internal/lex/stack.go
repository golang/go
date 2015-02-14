// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lex

import "text/scanner"

// A Stack is a stack of TokenReaders. As the top TokenReader hits EOF,
// it resumes reading the next one down.
type Stack struct {
	tr []TokenReader
}

// Push adds tr to the top (end) of the input stack. (Popping happens automatically.)
func (s *Stack) Push(tr TokenReader) {
	s.tr = append(s.tr, tr)
}

func (s *Stack) Next() ScanToken {
	tos := s.tr[len(s.tr)-1]
	tok := tos.Next()
	for tok == scanner.EOF && len(s.tr) > 1 {
		tos.Close()
		// Pop the topmost item from the stack and resume with the next one down.
		s.tr = s.tr[:len(s.tr)-1]
		tok = s.Next()
	}
	return tok
}

func (s *Stack) Text() string {
	return s.tr[len(s.tr)-1].Text()
}

func (s *Stack) File() string {
	return s.tr[len(s.tr)-1].File()
}

func (s *Stack) Line() int {
	return s.tr[len(s.tr)-1].Line()
}

func (s *Stack) Col() int {
	return s.tr[len(s.tr)-1].Col()
}

func (s *Stack) SetPos(line int, file string) {
	s.tr[len(s.tr)-1].SetPos(line, file)
}

func (s *Stack) Close() { // Unused.
}
