// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sfv

import (
	"errors"
	"strings"
)

// Params are an ordered map of key-value pairs that are associated with an item or an inner list.
//
// See https://httpwg.org/http-extensions/draft-ietf-httpbis-header-structure.html#param.
type Params struct {
	names  []string
	values map[string]interface{}
}

// ErrInvalidParameterFormat is returned when the format of a parameter is invalid.
var ErrInvalidParameterFormat = errors.New("invalid parameter format")

// ErrInvalidParameterValue is returned when a parameter key is invalid.
var ErrInvalidParameterValue = errors.New("invalid parameter value")

// NewParams creates a new ordered map.
func NewParams() *Params {
	p := Params{}
	p.names = []string{}
	p.values = map[string]interface{}{}

	return &p
}

// Get retrieves a parameter.
func (p *Params) Get(k string) (interface{}, bool) {
	v, ok := p.values[k]

	return v, ok
}

// Add appends a new parameter to the ordered list.
// If the key already exists, overwrite its value.
func (p *Params) Add(k string, v interface{}) {
	assertBareItem(v)

	if _, exists := p.values[k]; !exists {
		p.names = append(p.names, k)
	}

	p.values[k] = v
}

// Del removes a parameter from the ordered list.
func (p *Params) Del(key string) bool {
	if _, ok := p.values[key]; !ok {
		return false
	}

	for i, k := range p.names {
		if k == key {
			p.names = append(p.names[:i], p.names[i+1:]...)

			break
		}
	}

	delete(p.values, key)

	return true
}

// Names retrieves the list of parameter names in the appropriate order.
func (p *Params) Names() []string {
	return p.names
}

// marshalSFV serializes as defined in
// https://httpwg.org/http-extensions/draft-ietf-httpbis-header-structure.html#ser-params.
func (p *Params) marshalSFV(b *strings.Builder) error {
	for _, k := range p.names {
		if err := b.WriteByte(';'); err != nil {
			return err
		}

		if err := marshalKey(b, k); err != nil {
			return err
		}

		v := p.values[k]
		if v == true {
			continue
		}

		if err := b.WriteByte('='); err != nil {
			return err
		}

		if err := marshalBareItem(b, v); err != nil {
			return err
		}
	}

	return nil
}

// parseParams parses as defined in
// https://httpwg.org/http-extensions/draft-ietf-httpbis-header-structure.html#parse-param.
func parseParams(s *scanner) (*Params, error) {
	p := NewParams()

	for !s.eof() {
		if s.data[s.off] != ';' {
			break
		}
		s.off++
		s.scanWhileSp()

		k, err := parseKey(s)
		if err != nil {
			return nil, err
		}

		var i interface{}

		if !s.eof() && s.data[s.off] == '=' {
			s.off++

			i, err = parseBareItem(s)
			if err != nil {
				return nil, err
			}
		} else {
			i = true
		}

		p.Add(k, i)
	}

	return p, nil
}
