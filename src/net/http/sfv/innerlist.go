// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sfv

import (
	"errors"
	"strings"
)

// ErrInvalidInnerListFormat is returned when an inner list format is invalid.
var ErrInvalidInnerListFormat = errors.New("invalid inner list format")

// InnerList represents an inner list as defined in
// https://httpwg.org/http-extensions/draft-ietf-httpbis-header-structure.html#inner-list.
type InnerList struct {
	Items  []Item
	Params *Params
}

func (il InnerList) member() {
}

// marshalSFV serializes as defined in
// https://httpwg.org/http-extensions/draft-ietf-httpbis-header-structure.html#ser-innerlist.
func (il InnerList) marshalSFV(b *strings.Builder) error {
	if err := b.WriteByte('('); err != nil {
		return err
	}

	l := len(il.Items)
	for i := 0; i < l; i++ {
		if err := il.Items[i].marshalSFV(b); err != nil {
			return err
		}

		if i != l-1 {
			if err := b.WriteByte(' '); err != nil {
				return err
			}
		}
	}

	if err := b.WriteByte(')'); err != nil {
		return err
	}

	return il.Params.marshalSFV(b)
}

// parseInnerList parses as defined in
// https://httpwg.org/http-extensions/draft-ietf-httpbis-header-structure.html#parse-item-or-list.
func parseInnerList(s *scanner) (InnerList, error) {
	if s.eof() || s.data[s.off] != '(' {
		return InnerList{}, &UnmarshalError{s.off, ErrInvalidInnerListFormat}
	}
	s.off++

	il := InnerList{nil, nil}

	for !s.eof() {
		s.scanWhileSp()

		if s.eof() {
			return InnerList{}, &UnmarshalError{s.off, ErrInvalidInnerListFormat}
		}

		if s.data[s.off] == ')' {
			s.off++

			p, err := parseParams(s)
			if err != nil {
				return InnerList{}, err
			}

			il.Params = p

			return il, nil
		}

		i, err := parseItem(s)
		if err != nil {
			return InnerList{}, err
		}

		if s.eof() || (s.data[s.off] != ')' && s.data[s.off] != ' ') {
			return InnerList{}, &UnmarshalError{s.off, ErrInvalidInnerListFormat}
		}

		il.Items = append(il.Items, i)
	}

	return InnerList{}, &UnmarshalError{s.off, ErrInvalidInnerListFormat}
}
