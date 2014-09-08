// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bufio

// Exported for testing only.
import (
	"unicode/utf8"
)

var IsSpace = isSpace

func (s *Scanner) MaxTokenSize(n int) {
	if n < utf8.UTFMax || n > 1e9 {
		panic("bad max token size")
	}
	if n < len(s.buf) {
		s.buf = make([]byte, n)
	}
	s.maxTokenSize = n
}

// ErrOrEOF is like Err, but returns EOF. Used to test a corner case.
func (s *Scanner) ErrOrEOF() error {
	return s.err
}
