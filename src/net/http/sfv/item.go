// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sfv

import (
	"strings"
)

// Item is a bare value and associated parameters.
// See https://httpwg.org/http-extensions/draft-ietf-httpbis-header-structure.html#item.
type Item struct {
	Value  interface{}
	Params *Params
}

// NewItem returns a new Item.
func NewItem(v interface{}) Item {
	assertBareItem(v)

	return Item{v, NewParams()}
}

func (i Item) member() {
}

// marshalSFV serializes as defined in
// https://httpwg.org/http-extensions/draft-ietf-httpbis-header-structure.html#ser-item.
func (i Item) marshalSFV(b *strings.Builder) error {
	if i.Value == nil {
		return ErrInvalidBareItem
	}

	if err := marshalBareItem(b, i.Value); err != nil {
		return err
	}

	return i.Params.marshalSFV(b)
}

// UnmarshalItem parses an item as defined in
// https://httpwg.org/http-extensions/draft-ietf-httpbis-header-structure.html#parse-item.
func UnmarshalItem(v []string) (Item, error) {
	s := &scanner{
		data: strings.Join(v, ","),
	}

	s.scanWhileSp()

	sfv, err := parseItem(s)
	if err != nil {
		return Item{}, err
	}

	s.scanWhileSp()

	if !s.eof() {
		return Item{}, &UnmarshalError{off: s.off}
	}

	return sfv, nil
}

func parseItem(s *scanner) (Item, error) {
	bi, err := parseBareItem(s)
	if err != nil {
		return Item{}, err
	}

	p, err := parseParams(s)
	if err != nil {
		return Item{}, err
	}

	return Item{bi, p}, nil
}
