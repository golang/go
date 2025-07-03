// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jsonv2

package json

import (
	"archive/tar"
	"bytes"
	"errors"
	"io"
	"strings"
	"testing"

	"encoding/json/internal/jsonwire"
	"encoding/json/jsontext"
)

func TestSemanticError(t *testing.T) {
	tests := []struct {
		err  error
		want string
	}{{
		err:  &SemanticError{},
		want: `json: cannot handle`,
	}, {
		err:  &SemanticError{JSONKind: 'n'},
		want: `json: cannot handle JSON null`,
	}, {
		err:  &SemanticError{action: "unmarshal", JSONKind: 't'},
		want: `json: cannot unmarshal JSON boolean`,
	}, {
		err:  &SemanticError{action: "unmarshal", JSONKind: 'x'},
		want: `json: cannot unmarshal`, // invalid token kinds are ignored
	}, {
		err:  &SemanticError{action: "marshal", JSONKind: '"'},
		want: `json: cannot marshal JSON string`,
	}, {
		err:  &SemanticError{GoType: T[bool]()},
		want: `json: cannot handle Go bool`,
	}, {
		err:  &SemanticError{action: "marshal", GoType: T[int]()},
		want: `json: cannot marshal from Go int`,
	}, {
		err:  &SemanticError{action: "unmarshal", GoType: T[uint]()},
		want: `json: cannot unmarshal into Go uint`,
	}, {
		err:  &SemanticError{GoType: T[struct{ Alpha, Bravo, Charlie, Delta, Echo, Foxtrot, Golf, Hotel string }]()},
		want: `json: cannot handle Go struct`,
	}, {
		err:  &SemanticError{GoType: T[struct{ Alpha, Bravo, Charlie, Delta, Echo, Foxtrot, Golf, Hotel, x string }]()},
		want: `json: cannot handle Go v2.struct`,
	}, {
		err:  &SemanticError{JSONKind: '0', GoType: T[tar.Header]()},
		want: `json: cannot handle JSON number with Go tar.Header`,
	}, {
		err:  &SemanticError{action: "unmarshal", JSONKind: '0', JSONValue: jsontext.Value(`1e1000`), GoType: T[int]()},
		want: `json: cannot unmarshal JSON number 1e1000 into Go int`,
	}, {
		err:  &SemanticError{action: "marshal", JSONKind: '{', GoType: T[bytes.Buffer]()},
		want: `json: cannot marshal JSON object from Go bytes.Buffer`,
	}, {
		err:  &SemanticError{action: "unmarshal", JSONKind: ']', GoType: T[strings.Reader]()},
		want: `json: cannot unmarshal JSON array into Go strings.Reader`,
	}, {
		err:  &SemanticError{action: "unmarshal", JSONKind: '{', GoType: T[float64](), ByteOffset: 123},
		want: `json: cannot unmarshal JSON object into Go float64 after offset 123`,
	}, {
		err:  &SemanticError{action: "marshal", JSONKind: 'f', GoType: T[complex128](), ByteOffset: 123, JSONPointer: "/foo/2/bar/3"},
		want: `json: cannot marshal JSON boolean from Go complex128 within "/foo/2/bar/3"`,
	}, {
		err:  &SemanticError{action: "unmarshal", JSONKind: '}', GoType: T[io.Reader](), ByteOffset: 123, JSONPointer: "/foo/2/bar/3", Err: errors.New("some underlying error")},
		want: `json: cannot unmarshal JSON object into Go io.Reader within "/foo/2/bar/3": some underlying error`,
	}, {
		err:  &SemanticError{Err: errors.New("some underlying error")},
		want: `json: cannot handle: some underlying error`,
	}, {
		err:  &SemanticError{ByteOffset: 123},
		want: `json: cannot handle after offset 123`,
	}, {
		err:  &SemanticError{JSONPointer: "/foo/2/bar/3"},
		want: `json: cannot handle within "/foo/2/bar/3"`,
	}, {
		err:  &SemanticError{action: "unmarshal", JSONPointer: "/3", GoType: T[struct{ Fizz, Buzz string }](), Err: ErrUnknownName},
		want: `json: cannot unmarshal into Go struct { Fizz string; Buzz string }: unknown object member name "3"`,
	}, {
		err:  &SemanticError{action: "unmarshal", JSONPointer: "/foo/2/bar/3", GoType: T[struct{ Foo string }](), Err: ErrUnknownName},
		want: `json: cannot unmarshal into Go struct { Foo string }: unknown object member name "3" within "/foo/2/bar"`,
	}, {
		err:  &SemanticError{JSONPointer: "/foo/bar", ByteOffset: 16, GoType: T[string](), Err: &jsontext.SyntacticError{JSONPointer: "/foo/bar/baz", ByteOffset: 53, Err: jsonwire.ErrInvalidUTF8}},
		want: `json: cannot handle Go string: invalid UTF-8 within "/foo/bar/baz" after offset 53`,
	}, {
		err:  &SemanticError{JSONPointer: "/fizz/bar", ByteOffset: 16, GoType: T[string](), Err: &jsontext.SyntacticError{JSONPointer: "/foo/bar/baz", ByteOffset: 53, Err: jsonwire.ErrInvalidUTF8}},
		want: `json: cannot handle Go string within "/fizz/bar": invalid UTF-8 within "/foo/bar/baz" after offset 53`,
	}, {
		err:  &SemanticError{ByteOffset: 16, GoType: T[string](), Err: &jsontext.SyntacticError{JSONPointer: "/foo/bar/baz", ByteOffset: 53, Err: jsonwire.ErrInvalidUTF8}},
		want: `json: cannot handle Go string: invalid UTF-8 within "/foo/bar/baz" after offset 53`,
	}, {
		err:  &SemanticError{ByteOffset: 85, GoType: T[string](), Err: &jsontext.SyntacticError{JSONPointer: "/foo/bar/baz", ByteOffset: 53, Err: jsonwire.ErrInvalidUTF8}},
		want: `json: cannot handle Go string after offset 85: invalid UTF-8 within "/foo/bar/baz" after offset 53`,
	}}

	for _, tt := range tests {
		got := tt.err.Error()
		// Cleanup the error of non-deterministic rendering effects.
		if strings.HasPrefix(got, errorPrefix+"unable to ") {
			got = errorPrefix + "cannot " + strings.TrimPrefix(got, errorPrefix+"unable to ")
		}
		if got != tt.want {
			t.Errorf("%#v.Error mismatch:\ngot  %v\nwant %v", tt.err, got, tt.want)
		}
	}
}
