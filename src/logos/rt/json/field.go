// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package json

import (
	stdjson "encoding/json"
	"reflect"
)

// MarshalField is the per-field reflective fallback of generated MarshalJSON
// methods: it returns encoding/json.Marshal(v). Callers pass one field
// value and never the whole receiver, which would recurse into the
// generated method.
func MarshalField(v any) ([]byte, error) {
	return stdjson.Marshal(v)
}

// IsZero reports whether v equals the zero value of its type, which is
// reflect.Value.IsZero for every comparable type. Generated omitzero
// tests call it for comparable arrays and structs without an IsZero
// method.
func IsZero[T comparable](v T) bool {
	var zero T
	return v == zero
}

// IsNilPointer reports whether the dynamic value held in v is a nil
// pointer. Generated omitzero tests call it for a field whose static
// type is an interface declaring IsZero, so the dynamic type is known
// only at run time; encoding/json omits such a field without calling
// IsZero.
func IsNilPointer(v any) bool {
	rv := reflect.ValueOf(v)
	return rv.Kind() == reflect.Pointer && rv.IsNil()
}
