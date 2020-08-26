// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sfv

import (
	"errors"
	"io"
)

// ErrInvalidBooleanFormat is returned when a boolean format is invalid.
var ErrInvalidBooleanFormat = errors.New("invalid boolean format")

// marshalBoolean serializes as defined in
// https://httpwg.org/http-extensions/draft-ietf-httpbis-header-structure.html#ser-boolean.
func marshalBoolean(bd io.StringWriter, b bool) error {
	if b {
		_, err := bd.WriteString("?1")

		return err
	}

	_, err := bd.WriteString("?0")

	return err
}

// parseBoolean parses as defined in
// https://httpwg.org/http-extensions/draft-ietf-httpbis-header-structure.html#parse-boolean.
func parseBoolean(s *scanner) (bool, error) {
	if s.eof() || s.data[s.off] != '?' {
		return false, &UnmarshalError{s.off, ErrInvalidBooleanFormat}
	}
	s.off++

	if s.eof() {
		return false, &UnmarshalError{s.off, ErrInvalidBooleanFormat}
	}

	switch s.data[s.off] {
	case '0':
		s.off++

		return false, nil
	case '1':
		s.off++

		return true, nil
	}

	return false, &UnmarshalError{s.off, ErrInvalidBooleanFormat}
}
