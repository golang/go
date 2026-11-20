// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package httpsfv implements serializing and parsing
// of Structured Field Values for HTTP as defined in the draft-ietf-httpbis-header-structure Internet-Draft.
//
// Structured Field Values are either lists, dictionaries or items. Dedicated types are provided for all of them.
// Dedicated types are also used for tokens, parameters and inner lists.
// Other values are stored in native types:
//
//	int64, for integers
//	float64, for decimals
//	string, for strings
//	byte[], for byte sequences
//	bool, for booleans
//
// The specification is available at https://httpwg.org/http-extensions/draft-ietf-httpbis-header-structure.html.
package sfv

import (
	"strings"
)

// marshaler is the interface implemented by types that can marshal themselves into valid SFV.
type marshaler interface {
	marshalSFV(b *strings.Builder) error
}

// StructuredFieldValue represents a List, a Dictionary or an Item.
type StructuredFieldValue interface {
	marshaler
}

// Marshal returns the HTTP Structured Value serialization of v
// as defined in https://httpwg.org/http-extensions/draft-ietf-httpbis-header-structure.html#text-serialize.
//
// v must be a List, a Dictionary or an Item.
func Marshal(v StructuredFieldValue) (string, error) {
	var b strings.Builder
	if err := v.marshalSFV(&b); err != nil {
		return "", err
	}

	return b.String(), nil
}
