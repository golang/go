// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jsonv2

package jsontext

import (
	"encoding/json/internal/jsonflags"
	"encoding/json/internal/jsonwire"
)

// AppendQuote appends a double-quoted JSON string literal representing src
// to dst and returns the extended buffer.
// It uses the minimal string representation per RFC 8785, section 3.2.2.2.
// Invalid UTF-8 bytes are replaced with the Unicode replacement character
// and an error is returned at the end indicating the presence of invalid UTF-8.
// The dst must not overlap with the src.
func AppendQuote[Bytes ~[]byte | ~string](dst []byte, src Bytes) ([]byte, error) {
	dst, err := jsonwire.AppendQuote(dst, src, &jsonflags.Flags{})
	if err != nil {
		err = &SyntacticError{Err: err}
	}
	return dst, err
}

// AppendUnquote appends the decoded interpretation of src as a
// double-quoted JSON string literal to dst and returns the extended buffer.
// The input src must be a JSON string without any surrounding whitespace.
// Invalid UTF-8 bytes are replaced with the Unicode replacement character
// and an error is returned at the end indicating the presence of invalid UTF-8.
// Any trailing bytes after the JSON string literal results in an error.
// The dst must not overlap with the src.
func AppendUnquote[Bytes ~[]byte | ~string](dst []byte, src Bytes) ([]byte, error) {
	dst, err := jsonwire.AppendUnquote(dst, src)
	if err != nil {
		err = &SyntacticError{Err: err}
	}
	return dst, err
}
