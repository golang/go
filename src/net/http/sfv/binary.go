// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sfv

import (
	"encoding/base64"
	"errors"
	"strings"
)

// ErrInvalidBinaryFormat is returned when the binary format is invalid.
var ErrInvalidBinaryFormat = errors.New("invalid binary format")

// marshalBinary serializes as defined in
// https://httpwg.org/http-extensions/draft-ietf-httpbis-header-structure.html#ser-binary.
func marshalBinary(b *strings.Builder, bs []byte) error {
	if err := b.WriteByte(':'); err != nil {
		return err
	}

	buf := make([]byte, base64.StdEncoding.EncodedLen(len(bs)))
	base64.StdEncoding.Encode(buf, bs)

	if _, err := b.Write(buf); err != nil {
		return err
	}

	return b.WriteByte(':')
}

// parseBinary parses as defined in
// https://httpwg.org/http-extensions/draft-ietf-httpbis-header-structure.html#parse-binary.
func parseBinary(s *scanner) ([]byte, error) {
	if s.eof() || s.data[s.off] != ':' {
		return nil, &UnmarshalError{s.off, ErrInvalidBinaryFormat}
	}
	s.off++

	start := s.off

	for !s.eof() {
		c := s.data[s.off]
		if c == ':' {
			// base64decode
			decoded, err := base64.StdEncoding.DecodeString(s.data[start:s.off])
			if err != nil {
				return nil, &UnmarshalError{s.off, err}
			}
			s.off++

			return decoded, nil
		}

		if !isAlpha(c) && !isDigit(c) && c != '+' && c != '/' && c != '=' {
			return nil, &UnmarshalError{s.off, ErrInvalidBinaryFormat}
		}
		s.off++
	}

	return nil, &UnmarshalError{s.off, ErrInvalidBinaryFormat}
}
