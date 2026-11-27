// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sfv

import (
	"errors"
	"fmt"
	"io"
)

// isExtendedTchar checks if c is a valid token character as defined in the spec.
func isExtendedTchar(c byte) bool {
	if isAlpha(c) || isDigit(c) {
		return true
	}

	switch c {
	case '!', '#', '$', '%', '&', '\'', '*', '+', '-', '.', '^', '_', '`', '|', '~', ':', '/':
		return true
	}

	return false
}

// ErrInvalidTokenFormat is returned when a token format is invalid.
var ErrInvalidTokenFormat = errors.New("invalid token format")

// Token represents a token as defined in
// https://httpwg.org/http-extensions/draft-ietf-httpbis-header-structure.html#token.
// A specific type is used to distinguish tokens from strings.
type Token string

// marshalSFV serializes as defined in
// https://httpwg.org/http-extensions/draft-ietf-httpbis-header-structure.html#ser-token.
func (t Token) marshalSFV(b io.StringWriter) error {
	if len(t) == 0 {
		return fmt.Errorf("a token cannot be empty: %w", ErrInvalidTokenFormat)
	}

	if !isAlpha(t[0]) && t[0] != '*' {
		return fmt.Errorf("a token must start with an alpha character or *: %w", ErrInvalidTokenFormat)
	}

	for i := 1; i < len(t); i++ {
		if !isExtendedTchar(t[i]) {
			return fmt.Errorf("the character %c isn't allowed in a token: %w", t[i], ErrInvalidTokenFormat)
		}
	}

	_, err := b.WriteString(string(t))

	return err
}

// parseToken parses as defined in
// https://httpwg.org/http-extensions/draft-ietf-httpbis-header-structure.html#parse-token.
func parseToken(s *scanner) (Token, error) {
	if s.eof() || (!isAlpha(s.data[s.off]) && s.data[s.off] != '*') {
		return "", &UnmarshalError{s.off, ErrInvalidTokenFormat}
	}

	start := s.off
	s.off++

	for !s.eof() {
		if !isExtendedTchar(s.data[s.off]) {
			break
		}
		s.off++
	}

	return Token(s.data[start:s.off]), nil
}
