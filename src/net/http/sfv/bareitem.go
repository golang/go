// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sfv

import (
	"errors"
	"fmt"
	"reflect"
	"strings"
)

// ErrInvalidBareItem is returned when a bare item is invalid.
var ErrInvalidBareItem = errors.New(
	"invalid bare item type (allowed types are bool, string, int64, float64, []byte and Token)",
)

// assertBareItem asserts that v is a valid bare item
// according to https://httpwg.org/http-extensions/draft-ietf-httpbis-header-structure.html#item.
//
// v can be either:
//
// * an integer (Section 3.3.1.)
// * a decimal (Section 3.3.2.)
// * a string (Section 3.3.3.)
// * a token (Section 3.3.4.)
// * a byte sequence (Section 3.3.5.)
// * a boolean (Section 3.3.6.)
func assertBareItem(v interface{}) {
	switch v.(type) {
	case bool,
		string,
		int,
		int8,
		int16,
		int32,
		int64,
		uint,
		uint8,
		uint16,
		uint32,
		uint64,
		float32,
		float64,
		[]byte,
		Token:
		return
	default:
		panic(fmt.Errorf("%w: got %s", ErrInvalidBareItem, reflect.TypeOf(v)))
	}
}

// marshalBareItem serializes as defined in
// https://httpwg.org/http-extensions/draft-ietf-httpbis-header-structure.html#ser-bare-item.
func marshalBareItem(b *strings.Builder, v interface{}) error {
	switch v := v.(type) {
	case bool:
		return marshalBoolean(b, v)
	case string:
		return marshalString(b, v)
	case int64:
		return marshalInteger(b, v)
	case int, int8, int16, int32:
		return marshalInteger(b, reflect.ValueOf(v).Int())
	case uint, uint8, uint16, uint32, uint64:
		// Casting an uint64 to an int64 is possible because the maximum allowed value is 999,999,999,999,999
		return marshalInteger(b, int64(reflect.ValueOf(v).Uint()))
	case float32, float64:
		return marshalDecimal(b, v.(float64))
	case []byte:
		return marshalBinary(b, v)
	case Token:
		return v.marshalSFV(b)
	default:
		panic(ErrInvalidBareItem)
	}
}

// parseBareItem parses as defined in
// https://httpwg.org/http-extensions/draft-ietf-httpbis-header-structure.html#parse-bare-item.
func parseBareItem(s *scanner) (interface{}, error) {
	if s.eof() {
		return nil, &UnmarshalError{s.off, ErrUnexpectedEndOfString}
	}

	c := s.data[s.off]
	switch c {
	case '"':
		return parseString(s)
	case '?':
		return parseBoolean(s)
	case '*':
		return parseToken(s)
	case ':':
		return parseBinary(s)
	case '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9':
		return parseNumber(s)
	default:
		if isAlpha(c) {
			return parseToken(s)
		}

		return nil, &UnmarshalError{s.off, ErrUnrecognizedCharacter}
	}
}
