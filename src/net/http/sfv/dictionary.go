// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sfv

import (
	"errors"
	"strings"
)

// Dictionary is an ordered map of name-value pairs.
// See https://httpwg.org/http-extensions/draft-ietf-httpbis-header-structure.html#dictionary
// Values can be:
//   * Item (Section 3.3.)
//   * Inner List (Section 3.1.1.)
type Dictionary struct {
	names  []string
	values map[string]Member
}

// ErrInvalidDictionaryFormat is returned when a dictionary value is invalid.
var ErrInvalidDictionaryFormat = errors.New("invalid dictionary format")

// NewDictionary creates a new ordered map.
func NewDictionary() *Dictionary {
	d := Dictionary{}
	d.names = []string{}
	d.values = map[string]Member{}

	return &d
}

// Get retrieves a member.
func (d *Dictionary) Get(k string) (Member, bool) {
	v, ok := d.values[k]

	return v, ok
}

// Add appends a new member to the ordered list.
func (d *Dictionary) Add(k string, v Member) {
	if _, exists := d.values[k]; !exists {
		d.names = append(d.names, k)
	}

	d.values[k] = v
}

// Del removes a member from the ordered list.
func (d *Dictionary) Del(key string) bool {
	if _, ok := d.values[key]; !ok {
		return false
	}

	for i, k := range d.names {
		if k == key {
			d.names = append(d.names[:i], d.names[i+1:]...)

			break
		}
	}

	delete(d.values, key)

	return true
}

// Names retrieves the list of member names in the appropriate order.
func (d *Dictionary) Names() []string {
	return d.names
}

func (d *Dictionary) marshalSFV(b *strings.Builder) error {
	last := len(d.names) - 1

	for m, k := range d.names {
		if err := marshalKey(b, k); err != nil {
			return err
		}

		v := d.values[k]

		if item, ok := v.(Item); ok && item.Value == true {
			if err := item.Params.marshalSFV(b); err != nil {
				return err
			}
		} else {
			if err := b.WriteByte('='); err != nil {
				return err
			}
			if err := v.marshalSFV(b); err != nil {
				return err
			}
		}

		if m != last {
			if _, err := b.WriteString(", "); err != nil {
				return err
			}
		}
	}

	return nil
}

// UnmarshalDictionary parses a dictionary as defined in
// https://httpwg.org/http-extensions/draft-ietf-httpbis-header-structure.html#parse-dictionary.
func UnmarshalDictionary(v []string) (*Dictionary, error) {
	s := &scanner{
		data: strings.Join(v, ","),
	}

	s.scanWhileSp()

	sfv, err := parseDictionary(s)
	if err != nil {
		return sfv, err
	}

	return sfv, nil
}

func parseDictionary(s *scanner) (*Dictionary, error) {
	d := NewDictionary()

	for !s.eof() {
		k, err := parseKey(s)
		if err != nil {
			return nil, err
		}

		var m Member

		if !s.eof() && s.data[s.off] == '=' {
			s.off++
			m, err = parseItemOrInnerList(s)

			if err != nil {
				return nil, err
			}
		} else {
			p, err := parseParams(s)
			if err != nil {
				return nil, err
			}
			m = Item{true, p}
		}

		d.Add(k, m)
		s.scanWhileOWS()

		if s.eof() {
			return d, nil
		}

		if s.data[s.off] != ',' {
			return nil, &UnmarshalError{s.off, ErrInvalidDictionaryFormat}
		}
		s.off++

		s.scanWhileOWS()

		if s.eof() {
			// there is a trailing comma
			return nil, &UnmarshalError{s.off, ErrInvalidDictionaryFormat}
		}
	}

	return d, nil
}
