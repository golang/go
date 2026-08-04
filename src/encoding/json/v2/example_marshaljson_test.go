// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jsonv2

package json_test

import (
	"fmt"
	"log"
	"maps"
	"slices"

	"encoding/json/jsontext"
	"encoding/json/v2"
)

// IntSet is a simple set of numbers, implemented using a map.
type IntSet map[int]struct{}

// MarshalJSONTo encodes s as a JSON array into enc.
func (s *IntSet) MarshalJSONTo(enc *jsontext.Encoder) error {
	// Encode the set as a JSON array.
	if err := enc.WriteToken(jsontext.BeginArray); err != nil {
		return err
	}

	keys := maps.Keys(*s)

	// Go map iteration order is non-deterministic. If the caller has
	// requested deterministic output then we must sort the entries.
	deterministic, _ := json.GetOption(enc.Options(), json.Deterministic)
	if deterministic {
		keys = slices.Values(slices.Sorted(keys))
	}

	for k := range keys {
		// Call MarshalEncode instead of Write methods on Encoder so
		// that the "json" package can automatically handle any options
		// that may be relevant to the representation of k.
		// In this case, StringifyNumbers may affect whether k is
		// quoted or not.
		if err := json.MarshalEncode(enc, k); err != nil {
			return err
		}
	}

	if err := enc.WriteToken(jsontext.EndArray); err != nil {
		return err
	}
	return nil
}

// UnmarshalJSONFrom decodes a JSON array from dec into s.
func (s *IntSet) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	// Consume start of array.
	if k := dec.PeekKind(); k != '[' {
		// The [json] package automatically populates relevant fields
		// in a [json.SemanticError] to provide additional context.
		return &json.SemanticError{JSONKind: k}
	}
	if _, err := dec.ReadToken(); err != nil {
		return err
	}

	for dec.PeekKind() != ']' {
		var v int
		// See comment in MarshalJSONTo.
		if err := json.UnmarshalDecode(dec, &v); err != nil {
			return err
		}
		if *s == nil {
			*s = IntSet{}
		}
		(*s)[v] = struct{}{}
	}

	// Consume end of array.
	if _, err := dec.ReadToken(); err != nil {
		return err
	}
	return nil
}

// Custom types may define custom marshal behavior with [MarshalerTo].
func ExampleMarshalerTo() {
	set := IntSet{
		1: {},
		2: {},
		3: {},
	}

	b, err := json.Marshal(&set, json.Deterministic(true))
	if err != nil {
		log.Fatal(err)
	}

	// Indent output for readability.
	v := jsontext.Value(b)
	v.Indent()
	fmt.Println(string(v))

	// Output:
	// [
	// 	1,
	// 	2,
	// 	3
	// ]
}

// Custom types may define custom unmarshal behavior with [UnmarshalerFrom].
func ExampleUnmarshalerFrom() {
	s := "[1,2,3]"

	var set IntSet
	err := json.Unmarshal([]byte(s), &set)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(set)

	// Output:
	// map[1:{} 2:{} 3:{}]
}
