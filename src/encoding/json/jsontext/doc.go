// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jsonv2

// Package jsontext implements syntactic processing of JSON
// as specified in RFC 4627, RFC 7159, RFC 7493, RFC 8259, and RFC 8785.
// JSON is a simple data interchange format that can represent
// primitive data types such as booleans, strings, and numbers,
// in addition to structured data types such as objects and arrays.
//
// This package (encoding/json/jsontext) is experimental,
// and not subject to the Go 1 compatibility promise.
// It only exists when building with the GOEXPERIMENT=jsonv2 environment variable set.
// Most users should use [encoding/json].
//
// The [Encoder] and [Decoder] types are used to encode or decode
// a stream of JSON tokens or values.
//
// # Tokens and Values
//
// A JSON token refers to the basic structural elements of JSON:
//
//   - a JSON literal (i.e., null, true, or false)
//   - a JSON string (e.g., "hello, world!")
//   - a JSON number (e.g., 123.456)
//   - a begin or end delimiter for a JSON object (i.e., '{' or '}')
//   - a begin or end delimiter for a JSON array (i.e., '[' or ']')
//
// A JSON token is represented by the [Token] type in Go. Technically,
// there are two additional structural characters (i.e., ':' and ','),
// but there is no [Token] representation for them since their presence
// can be inferred by the structure of the JSON grammar itself.
// For example, there must always be an implicit colon between
// the name and value of a JSON object member.
//
// A JSON value refers to a complete unit of JSON data:
//
//   - a JSON literal, string, or number
//   - a JSON object (e.g., `{"name":"value"}`)
//   - a JSON array (e.g., `[1,2,3,]`)
//
// A JSON value is represented by the [Value] type in Go and is a []byte
// containing the raw textual representation of the value. There is some overlap
// between tokens and values as both contain literals, strings, and numbers.
// However, only a value can represent the entirety of a JSON object or array.
//
// The [Encoder] and [Decoder] types contain methods to read or write the next
// [Token] or [Value] in a sequence. They maintain a state machine to validate
// whether the sequence of JSON tokens and/or values produces a valid JSON.
// [Options] may be passed to the [NewEncoder] or [NewDecoder] constructors
// to configure the syntactic behavior of encoding and decoding.
//
// # Terminology
//
// The terms "encode" and "decode" are used for syntactic functionality
// that is concerned with processing JSON based on its grammar, and
// the terms "marshal" and "unmarshal" are used for semantic functionality
// that determines the meaning of JSON values as Go values and vice-versa.
// This package (i.e., [jsontext]) deals with JSON at a syntactic layer,
// while [encoding/json/v2] deals with JSON at a semantic layer.
// The goal is to provide a clear distinction between functionality that
// is purely concerned with encoding versus that of marshaling.
// For example, one can directly encode a stream of JSON tokens without
// needing to marshal a concrete Go value representing them.
// Similarly, one can decode a stream of JSON tokens without
// needing to unmarshal them into a concrete Go value.
//
// This package uses JSON terminology when discussing JSON, which may differ
// from related concepts in Go or elsewhere in computing literature.
//
//   - a JSON "object" refers to an unordered collection of name/value members.
//   - a JSON "array" refers to an ordered sequence of elements.
//   - a JSON "value" refers to either a literal (i.e., null, false, or true),
//     string, number, object, or array.
//
// See RFC 8259 for more information.
//
// # Specifications
//
// Relevant specifications include RFC 4627, RFC 7159, RFC 7493, RFC 8259,
// and RFC 8785. Each RFC is generally a stricter subset of another RFC.
// In increasing order of strictness:
//
//   - RFC 4627 and RFC 7159 do not require (but recommend) the use of UTF-8
//     and also do not require (but recommend) that object names be unique.
//   - RFC 8259 requires the use of UTF-8,
//     but does not require (but recommends) that object names be unique.
//   - RFC 7493 requires the use of UTF-8
//     and also requires that object names be unique.
//   - RFC 8785 defines a canonical representation. It requires the use of UTF-8
//     and also requires that object names be unique and in a specific ordering.
//     It specifies exactly how strings and numbers must be formatted.
//
// The primary difference between RFC 4627 and RFC 7159 is that the former
// restricted top-level values to only JSON objects and arrays, while
// RFC 7159 and subsequent RFCs permit top-level values to additionally be
// JSON nulls, booleans, strings, or numbers.
//
// By default, this package operates on RFC 7493, but can be configured
// to operate according to the other RFC specifications.
// RFC 7493 is a stricter subset of RFC 8259 and fully compliant with it.
// In particular, it makes specific choices about behavior that RFC 8259
// leaves as undefined in order to ensure greater interoperability.
package jsontext

// requireKeyedLiterals can be embedded in a struct to require keyed literals.
type requireKeyedLiterals struct{}

// nonComparable can be embedded in a struct to prevent comparability.
type nonComparable [0]func()
