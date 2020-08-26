// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sfv

import (
	"errors"
	"strings"
)

// ErrInvalidListFormat is returned when the format of a list is invalid.
var ErrInvalidListFormat = errors.New("invalid list format")

// List contains items an inner lists.
//
// See https://httpwg.org/http-extensions/draft-ietf-httpbis-header-structure.html#list
type List []Member

// marshalSFV serializes as defined in
// https://httpwg.org/http-extensions/draft-ietf-httpbis-header-structure.html#ser-list.
func (l List) marshalSFV(b *strings.Builder) error {
	s := len(l)
	for i := 0; i < s; i++ {
		if err := l[i].marshalSFV(b); err != nil {
			return err
		}

		if i != s-1 {
			if _, err := b.WriteString(", "); err != nil {
				return err
			}
		}
	}

	return nil
}

// UnmarshalList parses a list as defined in
// https://httpwg.org/http-extensions/draft-ietf-httpbis-header-structure.html#parse-list.
func UnmarshalList(v []string) (List, error) {
	s := &scanner{
		data: strings.Join(v, ","),
	}

	s.scanWhileSp()

	sfv, err := parseList(s)
	if err != nil {
		return List{}, err
	}

	return sfv, nil
}

// parseList parses as defined in
// https://httpwg.org/http-extensions/draft-ietf-httpbis-header-structure.html#parse-list.
func parseList(s *scanner) (List, error) {
	var l List

	for !s.eof() {
		m, err := parseItemOrInnerList(s)
		if err != nil {
			return nil, err
		}

		l = append(l, m)

		s.scanWhileOWS()

		if s.eof() {
			return l, nil
		}

		if s.data[s.off] != ',' {
			return nil, &UnmarshalError{s.off, ErrInvalidListFormat}
		}
		s.off++

		s.scanWhileOWS()

		if s.eof() {
			// there is a trailing comma
			return nil, &UnmarshalError{s.off, ErrInvalidListFormat}
		}
	}

	return l, nil
}

// parseItemOrInnerList parses as defined in
// https://httpwg.org/http-extensions/draft-ietf-httpbis-header-structure.html#parse-item-or-list.
func parseItemOrInnerList(s *scanner) (Member, error) {
	if s.data[s.off] == '(' {
		return parseInnerList(s)
	}

	return parseItem(s)
}
