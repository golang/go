// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jsonv2

// Represents JSON data structure using native Go types: booleans, floats,
// strings, arrays, and maps.

package json

import (
	"cmp"
	"fmt"
	"reflect"
	"strconv"

	"encoding/json/internal/jsonwire"
	"encoding/json/jsontext"
	jsonv2 "encoding/json/v2"
)

// Unmarshal parses the JSON-encoded data and stores the result
// in the value pointed to by v. If v is nil or not a pointer,
// Unmarshal returns an [InvalidUnmarshalError].
//
// Unmarshal uses the inverse of the encodings that
// [Marshal] uses, allocating maps, slices, and pointers as necessary,
// with the following additional rules:
//
// To unmarshal JSON into a pointer, Unmarshal first handles the case of
// the JSON being the JSON literal null. In that case, Unmarshal sets
// the pointer to nil. Otherwise, Unmarshal unmarshals the JSON into
// the value pointed at by the pointer. If the pointer is nil, Unmarshal
// allocates a new value for it to point to.
//
// To unmarshal JSON into a value implementing [Unmarshaler],
// Unmarshal calls that value's [Unmarshaler.UnmarshalJSON] method, including
// when the input is a JSON null.
// Otherwise, if the value implements [encoding.TextUnmarshaler]
// and the input is a JSON quoted string, Unmarshal calls
// [encoding.TextUnmarshaler.UnmarshalText] with the unquoted form of the string.
//
// To unmarshal JSON into a struct, Unmarshal matches incoming object
// keys to the keys used by [Marshal] (either the struct field name or its tag),
// preferring an exact match but also accepting a case-insensitive match. By
// default, object keys which don't have a corresponding struct field are
// ignored (see [Decoder.DisallowUnknownFields] for an alternative).
//
// To unmarshal JSON into an interface value,
// Unmarshal stores one of these in the interface value:
//
//   - bool, for JSON booleans
//   - float64, for JSON numbers
//   - string, for JSON strings
//   - []any, for JSON arrays
//   - map[string]any, for JSON objects
//   - nil for JSON null
//
// To unmarshal a JSON array into a slice, Unmarshal resets the slice length
// to zero and then appends each element to the slice.
// As a special case, to unmarshal an empty JSON array into a slice,
// Unmarshal replaces the slice with a new empty slice.
//
// To unmarshal a JSON array into a Go array, Unmarshal decodes
// JSON array elements into corresponding Go array elements.
// If the Go array is smaller than the JSON array,
// the additional JSON array elements are discarded.
// If the JSON array is smaller than the Go array,
// the additional Go array elements are set to zero values.
//
// To unmarshal a JSON object into a map, Unmarshal first establishes a map to
// use. If the map is nil, Unmarshal allocates a new map. Otherwise Unmarshal
// reuses the existing map, keeping existing entries. Unmarshal then stores
// key-value pairs from the JSON object into the map. The map's key type must
// either be any string type, an integer, or implement [encoding.TextUnmarshaler].
//
// If the JSON-encoded data contain a syntax error, Unmarshal returns a [SyntaxError].
//
// If a JSON value is not appropriate for a given target type,
// or if a JSON number overflows the target type, Unmarshal
// skips that field and completes the unmarshaling as best it can.
// If no more serious errors are encountered, Unmarshal returns
// an [UnmarshalTypeError] describing the earliest such error. In any
// case, it's not guaranteed that all the remaining fields following
// the problematic one will be unmarshaled into the target object.
//
// The JSON null value unmarshals into an interface, map, pointer, or slice
// by setting that Go value to nil. Because null is often used in JSON to mean
// “not present,” unmarshaling a JSON null into any other Go type has no effect
// on the value and produces no error.
//
// When unmarshaling quoted strings, invalid UTF-8 or
// invalid UTF-16 surrogate pairs are not treated as an error.
// Instead, they are replaced by the Unicode replacement
// character U+FFFD.
func Unmarshal(data []byte, v any) error {
	return jsonv2.Unmarshal(data, v, DefaultOptionsV1())
}

// Unmarshaler is the interface implemented by types
// that can unmarshal a JSON description of themselves.
// The input can be assumed to be a valid encoding of
// a JSON value. UnmarshalJSON must copy the JSON data
// if it wishes to retain the data after returning.
type Unmarshaler = jsonv2.Unmarshaler

// An UnmarshalTypeError describes a JSON value that was
// not appropriate for a value of a specific Go type.
type UnmarshalTypeError struct {
	Value  string       // description of JSON value - "bool", "array", "number -5"
	Type   reflect.Type // type of Go value it could not be assigned to
	Offset int64        // error occurred after reading Offset bytes
	Struct string       // name of the root type containing the field
	Field  string       // the full path from root node to the value
	Err    error        // may be nil
}

func (e *UnmarshalTypeError) Error() string {
	s := "json: cannot unmarshal"
	if e.Value != "" {
		s += " JSON " + e.Value
	}
	s += " into"
	var preposition string
	if e.Field != "" {
		s += " " + e.Struct + "." + e.Field
		preposition = " of"
	}
	if e.Type != nil {
		s += preposition
		s += " Go type " + e.Type.String()
	}
	if e.Err != nil {
		s += ": " + e.Err.Error()
	}
	return s
}

func (e *UnmarshalTypeError) Unwrap() error {
	return e.Err
}

// An UnmarshalFieldError describes a JSON object key that
// led to an unexported (and therefore unwritable) struct field.
//
// Deprecated: No longer used; kept for compatibility.
type UnmarshalFieldError struct {
	Key   string
	Type  reflect.Type
	Field reflect.StructField
}

func (e *UnmarshalFieldError) Error() string {
	return "json: cannot unmarshal object key " + strconv.Quote(e.Key) + " into unexported field " + e.Field.Name + " of type " + e.Type.String()
}

// An InvalidUnmarshalError describes an invalid argument passed to [Unmarshal].
// (The argument to [Unmarshal] must be a non-nil pointer.)
type InvalidUnmarshalError struct {
	Type reflect.Type
}

func (e *InvalidUnmarshalError) Error() string {
	if e.Type == nil {
		return "json: Unmarshal(nil)"
	}

	if e.Type.Kind() != reflect.Pointer {
		return "json: Unmarshal(non-pointer " + e.Type.String() + ")"
	}
	return "json: Unmarshal(nil " + e.Type.String() + ")"
}

// A Number represents a JSON number literal.
type Number string

// String returns the literal text of the number.
func (n Number) String() string { return string(n) }

// Float64 returns the number as a float64.
func (n Number) Float64() (float64, error) {
	return strconv.ParseFloat(string(n), 64)
}

// Int64 returns the number as an int64.
func (n Number) Int64() (int64, error) {
	return strconv.ParseInt(string(n), 10, 64)
}

var numberType = reflect.TypeFor[Number]()

// MarshalJSONTo implements [jsonv2.MarshalerTo].
func (n Number) MarshalJSONTo(enc *jsontext.Encoder) error {
	opts := enc.Options()
	stringify, _ := jsonv2.GetOption(opts, jsonv2.StringifyNumbers)
	if k, n := enc.StackIndex(enc.StackDepth()); k == '{' && n%2 == 0 {
		stringify = true // expecting a JSON object name
	}
	n = cmp.Or(n, "0")
	var num []byte
	val := enc.UnusedBuffer()
	if stringify {
		val = append(val, '"')
		val = append(val, n...)
		val = append(val, '"')
		num = val[len(`"`) : len(val)-len(`"`)]
	} else {
		val = append(val, n...)
		num = val
	}
	if n, err := jsonwire.ConsumeNumber(num); n != len(num) || err != nil {
		return fmt.Errorf("cannot parse %q as JSON number: %w", val, strconv.ErrSyntax)
	}
	return enc.WriteValue(val)
}

// UnmarshalJSONFrom implements [jsonv2.UnmarshalerFrom].
func (n *Number) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	opts := dec.Options()
	stringify, _ := jsonv2.GetOption(opts, jsonv2.StringifyNumbers)
	if k, n := dec.StackIndex(dec.StackDepth()); k == '{' && n%2 == 0 {
		stringify = true // expecting a JSON object name
	}
	val, err := dec.ReadValue()
	if err != nil {
		return err
	}
	val0 := val
	k := val.Kind()
	switch k {
	case 'n':
		if legacy, _ := jsonv2.GetOption(opts, MergeWithLegacySemantics); !legacy {
			*n = ""
		}
		return nil
	case '"':
		verbatim := jsonwire.ConsumeSimpleString(val) == len(val)
		val = jsonwire.UnquoteMayCopy(val, verbatim)
		if n, err := jsonwire.ConsumeNumber(val); n != len(val) || err != nil {
			return &jsonv2.SemanticError{JSONKind: val0.Kind(), JSONValue: val0.Clone(), GoType: numberType, Err: strconv.ErrSyntax}
		}
		*n = Number(val)
		return nil
	case '0':
		if stringify {
			break
		}
		*n = Number(val)
		return nil
	}
	return &jsonv2.SemanticError{JSONKind: k, GoType: numberType}
}
