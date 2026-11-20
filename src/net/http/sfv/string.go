// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sfv

import (
	"errors"
	"io"
	"strings"
	"unicode"
)

// ErrInvalidStringFormat is returned when a string format is invalid.
var ErrInvalidStringFormat = errors.New("invalid string format")

// marshalSFV serializes as defined in
// https://httpwg.org/http-extensions/draft-ietf-httpbis-header-structure.html#ser-string.
func marshalString(b io.ByteWriter, s string) error {
	if err := b.WriteByte('"'); err != nil {
		return err
	}

	for i := 0; i < len(s); i++ {
		if s[i] <= '\u001F' || s[i] >= unicode.MaxASCII {
			return ErrInvalidStringFormat
		}

		switch s[i] {
		case '"', '\\':
			if err := b.WriteByte('\\'); err != nil {
				return err
			}
		}

		if err := b.WriteByte(s[i]); err != nil {
			return err
		}
	}

	if err := b.WriteByte('"'); err != nil {
		return err
	}

	return nil
}

// parseString parses as defined in
// https://httpwg.org/http-extensions/draft-ietf-httpbis-header-structure.html#parse-string.
func parseString(s *scanner) (string, error) {
	if s.eof() || s.data[s.off] != '"' {
		return "", &UnmarshalError{s.off, ErrInvalidStringFormat}
	}
	s.off++

	var b strings.Builder

	for !s.eof() {
		c := s.data[s.off]
		s.off++

		switch c {
		case '\\':
			if s.eof() {
				return "", &UnmarshalError{s.off, ErrInvalidStringFormat}
			}

			n := s.data[s.off]
			if n != '"' && n != '\\' {
				return "", &UnmarshalError{s.off, ErrInvalidStringFormat}
			}
			s.off++

			if err := b.WriteByte(n); err != nil {
				return "", err
			}

			continue
		case '"':
			return b.String(), nil
		default:
			if c <= '\u001F' || c >= unicode.MaxASCII {
				return "", &UnmarshalError{s.off, ErrInvalidStringFormat}
			}

			if err := b.WriteByte(c); err != nil {
				return "", err
			}
		}
	}

	return "", &UnmarshalError{s.off, ErrInvalidStringFormat}
}
