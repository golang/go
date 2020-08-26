// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sfv

import (
	"errors"
	"fmt"
	"io"
)

// ErrInvalidKeyFormat is returned when the format of a parameter or dictionary key is invalid.
var ErrInvalidKeyFormat = errors.New("invalid key format")

// isKeyChar checks if c is a valid key characters.
func isKeyChar(c byte) bool {
	if isLowerCaseAlpha(c) || isDigit(c) {
		return true
	}

	switch c {
	case '_', '-', '.', '*':
		return true
	}

	return false
}

// checkKey checks if the given value is a valid parameter key according to
// https://httpwg.org/http-extensions/draft-ietf-httpbis-header-structure.html#param.
func checkKey(k string) error {
	if len(k) == 0 {
		return fmt.Errorf("a key cannot be empty: %w", ErrInvalidKeyFormat)
	}

	if !isLowerCaseAlpha(k[0]) && k[0] != '*' {
		return fmt.Errorf("a key must start with a lower case alpha character or *: %w", ErrInvalidKeyFormat)
	}

	for i := 1; i < len(k); i++ {
		if !isKeyChar(k[i]) {
			return fmt.Errorf("the character %c isn't allowed in a key: %w", k[i], ErrInvalidKeyFormat)
		}
	}

	return nil
}

// marshalKey serializes as defined in
// https://httpwg.org/http-extensions/draft-ietf-httpbis-header-structure.html#ser-key.
func marshalKey(b io.StringWriter, k string) error {
	if err := checkKey(k); err != nil {
		return err
	}

	_, err := b.WriteString(k)

	return err
}

// parseKey parses as defined in
// https://httpwg.org/http-extensions/draft-ietf-httpbis-header-structure.html#parse-key.
func parseKey(s *scanner) (string, error) {
	if s.eof() {
		return "", &UnmarshalError{s.off, ErrInvalidKeyFormat}
	}

	c := s.data[s.off]
	if !isLowerCaseAlpha(c) && c != '*' {
		return "", &UnmarshalError{s.off, ErrInvalidKeyFormat}
	}

	start := s.off
	s.off++

	for !s.eof() {
		if !isKeyChar(s.data[s.off]) {
			break
		}
		s.off++
	}

	return s.data[start:s.off], nil
}
