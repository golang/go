// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sfv

import (
	"errors"
	"fmt"
)

// ErrUnexpectedEndOfString is returned when the end of string is unexpected.
var ErrUnexpectedEndOfString = errors.New("unexpected end of string")

// ErrUnrecognizedCharacter is returned when an unrecognized character in encountered.
var ErrUnrecognizedCharacter = errors.New("unrecognized character")

// UnmarshalError contains the underlying parsing error and the position at which it occurred.
type UnmarshalError struct {
	off int
	err error
}

func (e *UnmarshalError) Error() string {
	if e.err != nil {
		return fmt.Sprintf("%s: character %d", e.err, e.off)
	}

	return fmt.Sprintf("unmarshal error: character %d", e.off)
}

func (e *UnmarshalError) Unwrap() error {
	return e.err
}

type scanner struct {
	data string
	off  int
}

// scanWhileSp consumes spaces.
func (s *scanner) scanWhileSp() {
	for !s.eof() {
		if s.data[s.off] != ' ' {
			return
		}

		s.off++
	}
}

// scanWhileOWS consumes optional white space (OWS) characters.
func (s *scanner) scanWhileOWS() {
	for !s.eof() {
		c := s.data[s.off]
		if c != ' ' && c != '\t' {
			return
		}

		s.off++
	}
}

// eof returns true if the parser consumed all available characters.
func (s *scanner) eof() bool {
	return s.off == len(s.data)
}
