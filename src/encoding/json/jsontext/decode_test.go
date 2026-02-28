// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jsonv2

package jsontext

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"net"
	"path"
	"reflect"
	"slices"
	"strings"
	"testing"
	"testing/iotest"

	"encoding/json/internal/jsonflags"
	"encoding/json/internal/jsontest"
	"encoding/json/internal/jsonwire"
)

// equalTokens reports whether to sequences of tokens formats the same way.
func equalTokens(xs, ys []Token) bool {
	if len(xs) != len(ys) {
		return false
	}
	for i := range xs {
		if !(reflect.DeepEqual(xs[i], ys[i]) || xs[i].String() == ys[i].String()) {
			return false
		}
	}
	return true
}

// TestDecoder tests whether we can parse JSON with either tokens or raw values.
func TestDecoder(t *testing.T) {
	for _, td := range coderTestdata {
		for _, typeName := range []string{"Token", "Value", "TokenDelims"} {
			t.Run(path.Join(td.name.Name, typeName), func(t *testing.T) {
				testDecoder(t, td.name.Where, typeName, td)
			})
		}
	}
}
func testDecoder(t *testing.T, where jsontest.CasePos, typeName string, td coderTestdataEntry) {
	dec := NewDecoder(bytes.NewBufferString(td.in))
	switch typeName {
	case "Token":
		var tokens []Token
		var pointers []Pointer
		for {
			tok, err := dec.ReadToken()
			if err != nil {
				if err == io.EOF {
					break
				}
				t.Fatalf("%s: Decoder.ReadToken error: %v", where, err)
			}
			tokens = append(tokens, tok.Clone())
			if td.pointers != nil {
				pointers = append(pointers, dec.StackPointer())
			}
		}
		if !equalTokens(tokens, td.tokens) {
			t.Fatalf("%s: tokens mismatch:\ngot  %v\nwant %v", where, tokens, td.tokens)
		}
		if !slices.Equal(pointers, td.pointers) {
			t.Fatalf("%s: pointers mismatch:\ngot  %q\nwant %q", where, pointers, td.pointers)
		}
	case "Value":
		val, err := dec.ReadValue()
		if err != nil {
			t.Fatalf("%s: Decoder.ReadValue error: %v", where, err)
		}
		got := string(val)
		want := strings.TrimSpace(td.in)
		if got != want {
			t.Fatalf("%s: Decoder.ReadValue = %s, want %s", where, got, want)
		}
	case "TokenDelims":
		// Use ReadToken for object/array delimiters, ReadValue otherwise.
		var tokens []Token
	loop:
		for {
			switch dec.PeekKind() {
			case '{', '}', '[', ']':
				tok, err := dec.ReadToken()
				if err != nil {
					if err == io.EOF {
						break loop
					}
					t.Fatalf("%s: Decoder.ReadToken error: %v", where, err)
				}
				tokens = append(tokens, tok.Clone())
			default:
				val, err := dec.ReadValue()
				if err != nil {
					if err == io.EOF {
						break loop
					}
					t.Fatalf("%s: Decoder.ReadValue error: %v", where, err)
				}
				tokens = append(tokens, rawToken(string(val)))
			}
		}
		if !equalTokens(tokens, td.tokens) {
			t.Fatalf("%s: tokens mismatch:\ngot  %v\nwant %v", where, tokens, td.tokens)
		}
	}
}

// TestFaultyDecoder tests that temporary I/O errors are not fatal.
func TestFaultyDecoder(t *testing.T) {
	for _, td := range coderTestdata {
		for _, typeName := range []string{"Token", "Value"} {
			t.Run(path.Join(td.name.Name, typeName), func(t *testing.T) {
				testFaultyDecoder(t, td.name.Where, typeName, td)
			})
		}
	}
}
func testFaultyDecoder(t *testing.T, where jsontest.CasePos, typeName string, td coderTestdataEntry) {
	b := &FaultyBuffer{
		B:        []byte(td.in),
		MaxBytes: 1,
		MayError: io.ErrNoProgress,
	}

	// Read all the tokens.
	// If the underlying io.Reader is faulty, then Read may return
	// an error without changing the internal state machine.
	// In other words, I/O errors occur before syntactic errors.
	dec := NewDecoder(b)
	switch typeName {
	case "Token":
		var tokens []Token
		for {
			tok, err := dec.ReadToken()
			if err != nil {
				if err == io.EOF {
					break
				}
				if !errors.Is(err, io.ErrNoProgress) {
					t.Fatalf("%s: %d: Decoder.ReadToken error: %v", where, len(tokens), err)
				}
				continue
			}
			tokens = append(tokens, tok.Clone())
		}
		if !equalTokens(tokens, td.tokens) {
			t.Fatalf("%s: tokens mismatch:\ngot  %s\nwant %s", where, tokens, td.tokens)
		}
	case "Value":
		for {
			val, err := dec.ReadValue()
			if err != nil {
				if err == io.EOF {
					break
				}
				if !errors.Is(err, io.ErrNoProgress) {
					t.Fatalf("%s: Decoder.ReadValue error: %v", where, err)
				}
				continue
			}
			got := string(val)
			want := strings.TrimSpace(td.in)
			if got != want {
				t.Fatalf("%s: Decoder.ReadValue = %s, want %s", where, got, want)
			}
		}
	}
}

type decoderMethodCall struct {
	wantKind    Kind
	wantOut     tokOrVal
	wantErr     error
	wantPointer Pointer
}

var decoderErrorTestdata = []struct {
	name       jsontest.CaseName
	opts       []Options
	in         string
	calls      []decoderMethodCall
	wantOffset int
}{{
	name: jsontest.Name("InvalidStart"),
	in:   ` #`,
	calls: []decoderMethodCall{
		{0, zeroToken, newInvalidCharacterError("#", "at start of value").withPos(" ", ""), ""},
		{0, zeroValue, newInvalidCharacterError("#", "at start of value").withPos(" ", ""), ""},
	},
}, {
	name: jsontest.Name("StreamN0"),
	in:   ` `,
	calls: []decoderMethodCall{
		{0, zeroToken, io.EOF, ""},
		{0, zeroValue, io.EOF, ""},
	},
}, {
	name: jsontest.Name("StreamN1"),
	in:   ` null `,
	calls: []decoderMethodCall{
		{'n', Null, nil, ""},
		{0, zeroToken, io.EOF, ""},
		{0, zeroValue, io.EOF, ""},
	},
	wantOffset: len(` null`),
}, {
	name: jsontest.Name("StreamN2"),
	in:   ` nullnull `,
	calls: []decoderMethodCall{
		{'n', Null, nil, ""},
		{'n', Null, nil, ""},
		{0, zeroToken, io.EOF, ""},
		{0, zeroValue, io.EOF, ""},
	},
	wantOffset: len(` nullnull`),
}, {
	name: jsontest.Name("StreamN2/ExtraComma"), // stream is whitespace delimited, not comma delimited
	in:   ` null , null `,
	calls: []decoderMethodCall{
		{'n', Null, nil, ""},
		{0, zeroToken, newInvalidCharacterError(",", `at start of value`).withPos(` null `, ""), ""},
		{0, zeroValue, newInvalidCharacterError(",", `at start of value`).withPos(` null `, ""), ""},
	},
	wantOffset: len(` null`),
}, {
	name: jsontest.Name("TruncatedNull"),
	in:   `nul`,
	calls: []decoderMethodCall{
		{'n', zeroToken, E(io.ErrUnexpectedEOF).withPos(`nul`, ""), ""},
		{'n', zeroValue, E(io.ErrUnexpectedEOF).withPos(`nul`, ""), ""},
	},
}, {
	name: jsontest.Name("InvalidNull"),
	in:   `nulL`,
	calls: []decoderMethodCall{
		{'n', zeroToken, newInvalidCharacterError("L", `in literal null (expecting 'l')`).withPos(`nul`, ""), ""},
		{'n', zeroValue, newInvalidCharacterError("L", `in literal null (expecting 'l')`).withPos(`nul`, ""), ""},
	},
}, {
	name: jsontest.Name("TruncatedFalse"),
	in:   `fals`,
	calls: []decoderMethodCall{
		{'f', zeroToken, E(io.ErrUnexpectedEOF).withPos(`fals`, ""), ""},
		{'f', zeroValue, E(io.ErrUnexpectedEOF).withPos(`fals`, ""), ""},
	},
}, {
	name: jsontest.Name("InvalidFalse"),
	in:   `falsE`,
	calls: []decoderMethodCall{
		{'f', zeroToken, newInvalidCharacterError("E", `in literal false (expecting 'e')`).withPos(`fals`, ""), ""},
		{'f', zeroValue, newInvalidCharacterError("E", `in literal false (expecting 'e')`).withPos(`fals`, ""), ""},
	},
}, {
	name: jsontest.Name("TruncatedTrue"),
	in:   `tru`,
	calls: []decoderMethodCall{
		{'t', zeroToken, E(io.ErrUnexpectedEOF).withPos(`tru`, ""), ""},
		{'t', zeroValue, E(io.ErrUnexpectedEOF).withPos(`tru`, ""), ""},
	},
}, {
	name: jsontest.Name("InvalidTrue"),
	in:   `truE`,
	calls: []decoderMethodCall{
		{'t', zeroToken, newInvalidCharacterError("E", `in literal true (expecting 'e')`).withPos(`tru`, ""), ""},
		{'t', zeroValue, newInvalidCharacterError("E", `in literal true (expecting 'e')`).withPos(`tru`, ""), ""},
	},
}, {
	name: jsontest.Name("TruncatedString"),
	in:   `"start`,
	calls: []decoderMethodCall{
		{'"', zeroToken, E(io.ErrUnexpectedEOF).withPos(`"start`, ""), ""},
		{'"', zeroValue, E(io.ErrUnexpectedEOF).withPos(`"start`, ""), ""},
	},
}, {
	name: jsontest.Name("InvalidString"),
	in:   `"ok` + "\x00",
	calls: []decoderMethodCall{
		{'"', zeroToken, newInvalidCharacterError("\x00", `in string (expecting non-control character)`).withPos(`"ok`, ""), ""},
		{'"', zeroValue, newInvalidCharacterError("\x00", `in string (expecting non-control character)`).withPos(`"ok`, ""), ""},
	},
}, {
	name: jsontest.Name("ValidString/AllowInvalidUTF8/Token"),
	opts: []Options{AllowInvalidUTF8(true)},
	in:   "\"living\xde\xad\xbe\xef\"",
	calls: []decoderMethodCall{
		{'"', rawToken("\"living\xde\xad\xbe\xef\""), nil, ""},
	},
	wantOffset: len("\"living\xde\xad\xbe\xef\""),
}, {
	name: jsontest.Name("ValidString/AllowInvalidUTF8/Value"),
	opts: []Options{AllowInvalidUTF8(true)},
	in:   "\"living\xde\xad\xbe\xef\"",
	calls: []decoderMethodCall{
		{'"', Value("\"living\xde\xad\xbe\xef\""), nil, ""},
	},
	wantOffset: len("\"living\xde\xad\xbe\xef\""),
}, {
	name: jsontest.Name("InvalidString/RejectInvalidUTF8"),
	opts: []Options{AllowInvalidUTF8(false)},
	in:   "\"living\xde\xad\xbe\xef\"",
	calls: []decoderMethodCall{
		{'"', zeroToken, E(jsonwire.ErrInvalidUTF8).withPos("\"living\xde\xad", ""), ""},
		{'"', zeroValue, E(jsonwire.ErrInvalidUTF8).withPos("\"living\xde\xad", ""), ""},
	},
}, {
	name: jsontest.Name("TruncatedNumber"),
	in:   `0.`,
	calls: []decoderMethodCall{
		{'0', zeroToken, E(io.ErrUnexpectedEOF), ""},
		{'0', zeroValue, E(io.ErrUnexpectedEOF), ""},
	},
}, {
	name: jsontest.Name("InvalidNumber"),
	in:   `0.e`,
	calls: []decoderMethodCall{
		{'0', zeroToken, newInvalidCharacterError("e", "in number (expecting digit)").withPos(`0.`, ""), ""},
		{'0', zeroValue, newInvalidCharacterError("e", "in number (expecting digit)").withPos(`0.`, ""), ""},
	},
}, {
	name: jsontest.Name("TruncatedObject/AfterStart"),
	in:   `{`,
	calls: []decoderMethodCall{
		{'{', zeroValue, E(io.ErrUnexpectedEOF).withPos("{", ""), ""},
		{'{', BeginObject, nil, ""},
		{0, zeroToken, E(io.ErrUnexpectedEOF).withPos("{", ""), ""},
		{0, zeroValue, E(io.ErrUnexpectedEOF).withPos("{", ""), ""},
	},
	wantOffset: len(`{`),
}, {
	name: jsontest.Name("TruncatedObject/AfterName"),
	in:   `{"0"`,
	calls: []decoderMethodCall{
		{'{', zeroValue, E(io.ErrUnexpectedEOF).withPos(`{"0"`, "/0"), ""},
		{'{', BeginObject, nil, ""},
		{'"', String("0"), nil, ""},
		{0, zeroToken, E(io.ErrUnexpectedEOF).withPos(`{"0"`, "/0"), ""},
		{0, zeroValue, E(io.ErrUnexpectedEOF).withPos(`{"0"`, "/0"), ""},
	},
	wantOffset: len(`{"0"`),
}, {
	name: jsontest.Name("TruncatedObject/AfterColon"),
	in:   `{"0":`,
	calls: []decoderMethodCall{
		{'{', zeroValue, E(io.ErrUnexpectedEOF).withPos(`{"0":`, "/0"), ""},
		{'{', BeginObject, nil, ""},
		{'"', String("0"), nil, ""},
		{0, zeroToken, E(io.ErrUnexpectedEOF).withPos(`{"0":`, "/0"), ""},
		{0, zeroValue, E(io.ErrUnexpectedEOF).withPos(`{"0":`, "/0"), ""},
	},
	wantOffset: len(`{"0"`),
}, {
	name: jsontest.Name("TruncatedObject/AfterValue"),
	in:   `{"0":0`,
	calls: []decoderMethodCall{
		{'{', zeroValue, E(io.ErrUnexpectedEOF).withPos(`{"0":0`, ""), ""},
		{'{', BeginObject, nil, ""},
		{'"', String("0"), nil, ""},
		{'0', Uint(0), nil, ""},
		{0, zeroToken, E(io.ErrUnexpectedEOF).withPos(`{"0":0`, ""), ""},
		{0, zeroValue, E(io.ErrUnexpectedEOF).withPos(`{"0":0`, ""), ""},
	},
	wantOffset: len(`{"0":0`),
}, {
	name: jsontest.Name("TruncatedObject/AfterComma"),
	in:   `{"0":0,`,
	calls: []decoderMethodCall{
		{'{', zeroValue, E(io.ErrUnexpectedEOF).withPos(`{"0":0,`, ""), ""},
		{'{', BeginObject, nil, ""},
		{'"', String("0"), nil, ""},
		{'0', Uint(0), nil, ""},
		{0, zeroToken, E(io.ErrUnexpectedEOF).withPos(`{"0":0,`, ""), ""},
		{0, zeroValue, E(io.ErrUnexpectedEOF).withPos(`{"0":0,`, ""), ""},
	},
	wantOffset: len(`{"0":0`),
}, {
	name: jsontest.Name("InvalidObject/MissingColon"),
	in:   ` { "fizz" "buzz" } `,
	calls: []decoderMethodCall{
		{'{', zeroValue, newInvalidCharacterError("\"", "after object name (expecting ':')").withPos(` { "fizz" `, "/fizz"), ""},
		{'{', BeginObject, nil, ""},
		{'"', String("fizz"), nil, ""},
		{0, zeroToken, newInvalidCharacterError("\"", "after object name (expecting ':')").withPos(` { "fizz" `, "/fizz"), ""},
		{0, zeroValue, newInvalidCharacterError("\"", "after object name (expecting ':')").withPos(` { "fizz" `, "/fizz"), ""},
	},
	wantOffset: len(` { "fizz"`),
}, {
	name: jsontest.Name("InvalidObject/MissingColon/GotComma"),
	in:   ` { "fizz" , "buzz" } `,
	calls: []decoderMethodCall{
		{'{', zeroValue, newInvalidCharacterError(",", "after object name (expecting ':')").withPos(` { "fizz" `, "/fizz"), ""},
		{'{', BeginObject, nil, ""},
		{'"', String("fizz"), nil, ""},
		{0, zeroToken, newInvalidCharacterError(",", "after object name (expecting ':')").withPos(` { "fizz" `, "/fizz"), ""},
		{0, zeroValue, newInvalidCharacterError(",", "after object name (expecting ':')").withPos(` { "fizz" `, "/fizz"), ""},
	},
	wantOffset: len(` { "fizz"`),
}, {
	name: jsontest.Name("InvalidObject/MissingColon/GotHash"),
	in:   ` { "fizz" # "buzz" } `,
	calls: []decoderMethodCall{
		{'{', zeroValue, newInvalidCharacterError("#", "after object name (expecting ':')").withPos(` { "fizz" `, "/fizz"), ""},
		{'{', BeginObject, nil, ""},
		{'"', String("fizz"), nil, ""},
		{0, zeroToken, newInvalidCharacterError("#", "after object name (expecting ':')").withPos(` { "fizz" `, "/fizz"), ""},
		{0, zeroValue, newInvalidCharacterError("#", "after object name (expecting ':')").withPos(` { "fizz" `, "/fizz"), ""},
	},
	wantOffset: len(` { "fizz"`),
}, {
	name: jsontest.Name("InvalidObject/MissingComma"),
	in:   ` { "fizz" : "buzz" "gazz" } `,
	calls: []decoderMethodCall{
		{'{', zeroValue, newInvalidCharacterError("\"", "after object value (expecting ',' or '}')").withPos(` { "fizz" : "buzz" `, ""), ""},
		{'{', BeginObject, nil, ""},
		{'"', String("fizz"), nil, ""},
		{'"', String("buzz"), nil, ""},
		{0, zeroToken, newInvalidCharacterError("\"", "after object value (expecting ',' or '}')").withPos(` { "fizz" : "buzz" `, ""), ""},
		{0, zeroValue, newInvalidCharacterError("\"", "after object value (expecting ',' or '}')").withPos(` { "fizz" : "buzz" `, ""), ""},
	},
	wantOffset: len(` { "fizz" : "buzz"`),
}, {
	name: jsontest.Name("InvalidObject/MissingComma/GotColon"),
	in:   ` { "fizz" : "buzz" : "gazz" } `,
	calls: []decoderMethodCall{
		{'{', zeroValue, newInvalidCharacterError(":", "after object value (expecting ',' or '}')").withPos(` { "fizz" : "buzz" `, ""), ""},
		{'{', BeginObject, nil, ""},
		{'"', String("fizz"), nil, ""},
		{'"', String("buzz"), nil, ""},
		{0, zeroToken, newInvalidCharacterError(":", "after object value (expecting ',' or '}')").withPos(` { "fizz" : "buzz" `, ""), ""},
		{0, zeroValue, newInvalidCharacterError(":", "after object value (expecting ',' or '}')").withPos(` { "fizz" : "buzz" `, ""), ""},
	},
	wantOffset: len(` { "fizz" : "buzz"`),
}, {
	name: jsontest.Name("InvalidObject/MissingComma/GotHash"),
	in:   ` { "fizz" : "buzz" # "gazz" } `,
	calls: []decoderMethodCall{
		{'{', zeroValue, newInvalidCharacterError("#", "after object value (expecting ',' or '}')").withPos(` { "fizz" : "buzz" `, ""), ""},
		{'{', BeginObject, nil, ""},
		{'"', String("fizz"), nil, ""},
		{'"', String("buzz"), nil, ""},
		{0, zeroToken, newInvalidCharacterError("#", "after object value (expecting ',' or '}')").withPos(` { "fizz" : "buzz" `, ""), ""},
		{0, zeroValue, newInvalidCharacterError("#", "after object value (expecting ',' or '}')").withPos(` { "fizz" : "buzz" `, ""), ""},
	},
	wantOffset: len(` { "fizz" : "buzz"`),
}, {
	name: jsontest.Name("InvalidObject/ExtraComma/AfterStart"),
	in:   ` { , } `,
	calls: []decoderMethodCall{
		{'{', zeroValue, newInvalidCharacterError(",", `at start of string (expecting '"')`).withPos(` { `, ""), ""},
		{'{', BeginObject, nil, ""},
		{0, zeroToken, newInvalidCharacterError(",", `at start of value`).withPos(` { `, ""), ""},
		{0, zeroValue, newInvalidCharacterError(",", `at start of value`).withPos(` { `, ""), ""},
	},
	wantOffset: len(` {`),
}, {
	name: jsontest.Name("InvalidObject/ExtraComma/AfterValue"),
	in:   ` { "fizz" : "buzz" , } `,
	calls: []decoderMethodCall{
		{'{', zeroValue, newInvalidCharacterError("}", `at start of string (expecting '"')`).withPos(` { "fizz" : "buzz" , `, ""), ""},
		{'{', BeginObject, nil, ""},
		{'"', String("fizz"), nil, ""},
		{'"', String("buzz"), nil, ""},
		{0, zeroToken, newInvalidCharacterError(",", `at start of value`).withPos(` { "fizz" : "buzz" `, ""), ""},
		{0, zeroValue, newInvalidCharacterError(",", `at start of value`).withPos(` { "fizz" : "buzz" `, ""), ""},
	},
	wantOffset: len(` { "fizz" : "buzz"`),
}, {
	name: jsontest.Name("InvalidObject/InvalidName/GotNull"),
	in:   ` { null : null } `,
	calls: []decoderMethodCall{
		{'{', zeroValue, newInvalidCharacterError("n", "at start of string (expecting '\"')").withPos(` { `, ""), ""},
		{'{', BeginObject, nil, ""},
		{'n', zeroToken, E(ErrNonStringName).withPos(` { `, ""), ""},
		{'n', zeroValue, E(ErrNonStringName).withPos(` { `, ""), ""},
	},
	wantOffset: len(` {`),
}, {
	name: jsontest.Name("InvalidObject/InvalidName/GotFalse"),
	in:   ` { false : false } `,
	calls: []decoderMethodCall{
		{'{', zeroValue, newInvalidCharacterError("f", "at start of string (expecting '\"')").withPos(` { `, ""), ""},
		{'{', BeginObject, nil, ""},
		{'f', zeroToken, E(ErrNonStringName).withPos(` { `, ""), ""},
		{'f', zeroValue, E(ErrNonStringName).withPos(` { `, ""), ""},
	},
	wantOffset: len(` {`),
}, {
	name: jsontest.Name("InvalidObject/InvalidName/GotTrue"),
	in:   ` { true : true } `,
	calls: []decoderMethodCall{
		{'{', zeroValue, newInvalidCharacterError("t", "at start of string (expecting '\"')").withPos(` { `, ""), ""},
		{'{', BeginObject, nil, ""},
		{'t', zeroToken, E(ErrNonStringName).withPos(` { `, ""), ""},
		{'t', zeroValue, E(ErrNonStringName).withPos(` { `, ""), ""},
	},
	wantOffset: len(` {`),
}, {
	name: jsontest.Name("InvalidObject/InvalidName/GotNumber"),
	in:   ` { 0 : 0 } `,
	calls: []decoderMethodCall{
		{'{', zeroValue, newInvalidCharacterError("0", "at start of string (expecting '\"')").withPos(` { `, ""), ""},
		{'{', BeginObject, nil, ""},
		{'0', zeroToken, E(ErrNonStringName).withPos(` { `, ""), ""},
		{'0', zeroValue, E(ErrNonStringName).withPos(` { `, ""), ""},
	},
	wantOffset: len(` {`),
}, {
	name: jsontest.Name("InvalidObject/InvalidName/GotObject"),
	in:   ` { {} : {} } `,
	calls: []decoderMethodCall{
		{'{', zeroValue, newInvalidCharacterError("{", "at start of string (expecting '\"')").withPos(` { `, ""), ""},
		{'{', BeginObject, nil, ""},
		{'{', zeroToken, E(ErrNonStringName).withPos(` { `, ""), ""},
		{'{', zeroValue, E(ErrNonStringName).withPos(` { `, ""), ""},
	},
	wantOffset: len(` {`),
}, {
	name: jsontest.Name("InvalidObject/InvalidName/GotArray"),
	in:   ` { [] : [] } `,
	calls: []decoderMethodCall{
		{'{', zeroValue, newInvalidCharacterError("[", "at start of string (expecting '\"')").withPos(` { `, ""), ""},
		{'{', BeginObject, nil, ""},
		{'[', zeroToken, E(ErrNonStringName).withPos(` { `, ""), ""},
		{'[', zeroValue, E(ErrNonStringName).withPos(` { `, ""), ""},
	},
	wantOffset: len(` {`),
}, {
	name: jsontest.Name("InvalidObject/MismatchingDelim"),
	in:   ` { ] `,
	calls: []decoderMethodCall{
		{'{', zeroValue, newInvalidCharacterError("]", "at start of string (expecting '\"')").withPos(` { `, ""), ""},
		{'{', BeginObject, nil, ""},
		{']', zeroToken, newInvalidCharacterError("]", "at start of value").withPos(` { `, ""), ""},
		{']', zeroValue, newInvalidCharacterError("]", "at start of value").withPos(` { `, ""), ""},
	},
	wantOffset: len(` {`),
}, {
	name: jsontest.Name("ValidObject/InvalidValue"),
	in:   ` { } `,
	calls: []decoderMethodCall{
		{'{', BeginObject, nil, ""},
		{'}', zeroValue, newInvalidCharacterError("}", "at start of value").withPos(" { ", ""), ""},
	},
	wantOffset: len(` {`),
}, {
	name: jsontest.Name("ValidObject/UniqueNames"),
	in:   `{"0":0,"1":1} `,
	calls: []decoderMethodCall{
		{'{', BeginObject, nil, ""},
		{'"', String("0"), nil, ""},
		{'0', Uint(0), nil, ""},
		{'"', String("1"), nil, ""},
		{'0', Uint(1), nil, ""},
		{'}', EndObject, nil, ""},
	},
	wantOffset: len(`{"0":0,"1":1}`),
}, {
	name: jsontest.Name("ValidObject/DuplicateNames"),
	opts: []Options{AllowDuplicateNames(true)},
	in:   `{"0":0,"0":0} `,
	calls: []decoderMethodCall{
		{'{', BeginObject, nil, ""},
		{'"', String("0"), nil, ""},
		{'0', Uint(0), nil, ""},
		{'"', String("0"), nil, ""},
		{'0', Uint(0), nil, ""},
		{'}', EndObject, nil, ""},
	},
	wantOffset: len(`{"0":0,"0":0}`),
}, {
	name: jsontest.Name("InvalidObject/DuplicateNames"),
	in:   `{"X":{},"Y":{},"X":{}} `,
	calls: []decoderMethodCall{
		{'{', zeroValue, E(ErrDuplicateName).withPos(`{"X":{},"Y":{},`, "/X"), ""},
		{'{', BeginObject, nil, ""},
		{'"', String("X"), nil, ""},
		{'{', BeginObject, nil, ""},
		{'}', EndObject, nil, ""},
		{'"', String("Y"), nil, ""},
		{'{', BeginObject, nil, ""},
		{'}', EndObject, nil, ""},
		{'"', zeroToken, E(ErrDuplicateName).withPos(`{"X":{},"Y":{},`, "/X"), "/Y"},
		{'"', zeroValue, E(ErrDuplicateName).withPos(`{"0":{},"Y":{},`, "/X"), "/Y"},
	},
	wantOffset: len(`{"0":{},"1":{}`),
}, {
	name: jsontest.Name("TruncatedArray/AfterStart"),
	in:   `[`,
	calls: []decoderMethodCall{
		{'[', zeroValue, E(io.ErrUnexpectedEOF).withPos("[", ""), ""},
		{'[', BeginArray, nil, ""},
		{0, zeroToken, E(io.ErrUnexpectedEOF).withPos("[", ""), ""},
		{0, zeroValue, E(io.ErrUnexpectedEOF).withPos("[", ""), ""},
	},
	wantOffset: len(`[`),
}, {
	name: jsontest.Name("TruncatedArray/AfterValue"),
	in:   `[0`,
	calls: []decoderMethodCall{
		{'[', zeroValue, E(io.ErrUnexpectedEOF).withPos("[0", ""), ""},
		{'[', BeginArray, nil, ""},
		{'0', Uint(0), nil, ""},
		{0, zeroToken, E(io.ErrUnexpectedEOF).withPos("[0", ""), ""},
		{0, zeroValue, E(io.ErrUnexpectedEOF).withPos("[0", ""), ""},
	},
	wantOffset: len(`[0`),
}, {
	name: jsontest.Name("TruncatedArray/AfterComma"),
	in:   `[0,`,
	calls: []decoderMethodCall{
		{'[', zeroValue, E(io.ErrUnexpectedEOF).withPos("[0,", ""), ""},
		{'[', BeginArray, nil, ""},
		{'0', Uint(0), nil, ""},
		{0, zeroToken, E(io.ErrUnexpectedEOF).withPos("[0,", ""), ""},
		{0, zeroValue, E(io.ErrUnexpectedEOF).withPos("[0,", ""), ""},
	},
	wantOffset: len(`[0`),
}, {
	name: jsontest.Name("InvalidArray/MissingComma"),
	in:   ` [ "fizz" "buzz" ] `,
	calls: []decoderMethodCall{
		{'[', zeroValue, newInvalidCharacterError("\"", "after array element (expecting ',' or ']')").withPos(` [ "fizz" `, ""), ""},
		{'[', BeginArray, nil, ""},
		{'"', String("fizz"), nil, ""},
		{0, zeroToken, newInvalidCharacterError("\"", "after array element (expecting ',' or ']')").withPos(` [ "fizz" `, ""), ""},
		{0, zeroValue, newInvalidCharacterError("\"", "after array element (expecting ',' or ']')").withPos(` [ "fizz" `, ""), ""},
	},
	wantOffset: len(` [ "fizz"`),
}, {
	name: jsontest.Name("InvalidArray/MismatchingDelim"),
	in:   ` [ } `,
	calls: []decoderMethodCall{
		{'[', zeroValue, newInvalidCharacterError("}", "at start of value").withPos(` [ `, "/0"), ""},
		{'[', BeginArray, nil, ""},
		{'}', zeroToken, newInvalidCharacterError("}", "at start of value").withPos(` [ `, "/0"), ""},
		{'}', zeroValue, newInvalidCharacterError("}", "at start of value").withPos(` [ `, "/0"), ""},
	},
	wantOffset: len(` [`),
}, {
	name: jsontest.Name("ValidArray/InvalidValue"),
	in:   ` [ ] `,
	calls: []decoderMethodCall{
		{'[', BeginArray, nil, ""},
		{']', zeroValue, newInvalidCharacterError("]", "at start of value").withPos(" [ ", "/0"), ""},
	},
	wantOffset: len(` [`),
}, {
	name: jsontest.Name("InvalidDelim/AfterTopLevel"),
	in:   `"",`,
	calls: []decoderMethodCall{
		{'"', String(""), nil, ""},
		{0, zeroToken, newInvalidCharacterError(",", "at start of value").withPos(`""`, ""), ""},
		{0, zeroValue, newInvalidCharacterError(",", "at start of value").withPos(`""`, ""), ""},
	},
	wantOffset: len(`""`),
}, {
	name: jsontest.Name("InvalidDelim/AfterBeginObject"),
	in:   `{:`,
	calls: []decoderMethodCall{
		{'{', zeroValue, newInvalidCharacterError(":", `at start of string (expecting '"')`).withPos(`{`, ""), ""},
		{'{', BeginObject, nil, ""},
		{0, zeroToken, newInvalidCharacterError(":", "at start of value").withPos(`{`, ""), ""},
		{0, zeroValue, newInvalidCharacterError(":", "at start of value").withPos(`{`, ""), ""},
	},
	wantOffset: len(`{`),
}, {
	name: jsontest.Name("InvalidDelim/AfterObjectName"),
	in:   `{"",`,
	calls: []decoderMethodCall{
		{'{', zeroValue, newInvalidCharacterError(",", "after object name (expecting ':')").withPos(`{""`, "/"), ""},
		{'{', BeginObject, nil, ""},
		{'"', String(""), nil, ""},
		{0, zeroToken, newInvalidCharacterError(",", "after object name (expecting ':')").withPos(`{""`, "/"), ""},
		{0, zeroValue, newInvalidCharacterError(",", "after object name (expecting ':')").withPos(`{""`, "/"), ""},
	},
	wantOffset: len(`{""`),
}, {
	name: jsontest.Name("ValidDelim/AfterObjectName"),
	in:   `{"":`,
	calls: []decoderMethodCall{
		{'{', zeroValue, E(io.ErrUnexpectedEOF).withPos(`{"":`, "/"), ""},
		{'{', BeginObject, nil, ""},
		{'"', String(""), nil, ""},
		{0, zeroToken, E(io.ErrUnexpectedEOF).withPos(`{"":`, "/"), ""},
		{0, zeroValue, E(io.ErrUnexpectedEOF).withPos(`{"":`, "/"), ""},
	},
	wantOffset: len(`{""`),
}, {
	name: jsontest.Name("InvalidDelim/AfterObjectValue"),
	in:   `{"":"":`,
	calls: []decoderMethodCall{
		{'{', zeroValue, newInvalidCharacterError(":", "after object value (expecting ',' or '}')").withPos(`{"":""`, ""), ""},
		{'{', BeginObject, nil, ""},
		{'"', String(""), nil, ""},
		{'"', String(""), nil, ""},
		{0, zeroToken, newInvalidCharacterError(":", "after object value (expecting ',' or '}')").withPos(`{"":""`, ""), ""},
		{0, zeroValue, newInvalidCharacterError(":", "after object value (expecting ',' or '}')").withPos(`{"":""`, ""), ""},
	},
	wantOffset: len(`{"":""`),
}, {
	name: jsontest.Name("ValidDelim/AfterObjectValue"),
	in:   `{"":"",`,
	calls: []decoderMethodCall{
		{'{', zeroValue, E(io.ErrUnexpectedEOF).withPos(`{"":"",`, ""), ""},
		{'{', BeginObject, nil, ""},
		{'"', String(""), nil, ""},
		{'"', String(""), nil, ""},
		{0, zeroToken, E(io.ErrUnexpectedEOF).withPos(`{"":"",`, ""), ""},
		{0, zeroValue, E(io.ErrUnexpectedEOF).withPos(`{"":"",`, ""), ""},
	},
	wantOffset: len(`{"":""`),
}, {
	name: jsontest.Name("InvalidDelim/AfterBeginArray"),
	in:   `[,`,
	calls: []decoderMethodCall{
		{'[', zeroValue, newInvalidCharacterError(",", "at start of value").withPos(`[`, "/0"), ""},
		{'[', BeginArray, nil, ""},
		{0, zeroToken, newInvalidCharacterError(",", "at start of value").withPos(`[`, ""), ""},
		{0, zeroValue, newInvalidCharacterError(",", "at start of value").withPos(`[`, ""), ""},
	},
	wantOffset: len(`[`),
}, {
	name: jsontest.Name("InvalidDelim/AfterArrayValue"),
	in:   `["":`,
	calls: []decoderMethodCall{
		{'[', zeroValue, newInvalidCharacterError(":", "after array element (expecting ',' or ']')").withPos(`[""`, ""), ""},
		{'[', BeginArray, nil, ""},
		{'"', String(""), nil, ""},
		{0, zeroToken, newInvalidCharacterError(":", "after array element (expecting ',' or ']')").withPos(`[""`, ""), ""},
		{0, zeroValue, newInvalidCharacterError(":", "after array element (expecting ',' or ']')").withPos(`[""`, ""), ""},
	},
	wantOffset: len(`[""`),
}, {
	name: jsontest.Name("ValidDelim/AfterArrayValue"),
	in:   `["",`,
	calls: []decoderMethodCall{
		{'[', zeroValue, E(io.ErrUnexpectedEOF).withPos(`["",`, ""), ""},
		{'[', BeginArray, nil, ""},
		{'"', String(""), nil, ""},
		{0, zeroToken, E(io.ErrUnexpectedEOF).withPos(`["",`, ""), ""},
		{0, zeroValue, E(io.ErrUnexpectedEOF).withPos(`["",`, ""), ""},
	},
	wantOffset: len(`[""`),
}, {
	name: jsontest.Name("ErrorPosition"),
	in:   ` "a` + "\xff" + `0" `,
	calls: []decoderMethodCall{
		{'"', zeroValue, E(jsonwire.ErrInvalidUTF8).withPos(` "a`, ""), ""},
		{'"', zeroToken, E(jsonwire.ErrInvalidUTF8).withPos(` "a`, ""), ""},
	},
}, {
	name: jsontest.Name("ErrorPosition/0"),
	in:   ` [ "a` + "\xff" + `1" ] `,
	calls: []decoderMethodCall{
		{'[', zeroValue, E(jsonwire.ErrInvalidUTF8).withPos(` [ "a`, "/0"), ""},
		{'[', BeginArray, nil, ""},
		{'"', zeroValue, E(jsonwire.ErrInvalidUTF8).withPos(` [ "a`, "/0"), ""},
		{'"', zeroToken, E(jsonwire.ErrInvalidUTF8).withPos(` [ "a`, "/0"), ""},
	},
	wantOffset: len(` [`),
}, {
	name: jsontest.Name("ErrorPosition/1"),
	in:   ` [ "a1" , "b` + "\xff" + `1" ] `,
	calls: []decoderMethodCall{
		{'[', zeroValue, E(jsonwire.ErrInvalidUTF8).withPos(` [ "a1" , "b`, "/1"), ""},
		{'[', BeginArray, nil, ""},
		{'"', String("a1"), nil, ""},
		{'"', zeroValue, E(jsonwire.ErrInvalidUTF8).withPos(` [ "a1" , "b`, "/1"), ""},
		{'"', zeroToken, E(jsonwire.ErrInvalidUTF8).withPos(` [ "a1" , "b`, "/1"), ""},
	},
	wantOffset: len(` [ "a1"`),
}, {
	name: jsontest.Name("ErrorPosition/0/0"),
	in:   ` [ [ "a` + "\xff" + `2" ] ] `,
	calls: []decoderMethodCall{
		{'[', zeroValue, E(jsonwire.ErrInvalidUTF8).withPos(` [ [ "a`, "/0/0"), ""},
		{'[', BeginArray, nil, ""},
		{'[', zeroValue, E(jsonwire.ErrInvalidUTF8).withPos(` [ [ "a`, "/0/0"), ""},
		{'[', BeginArray, nil, "/0"},
		{'"', zeroValue, E(jsonwire.ErrInvalidUTF8).withPos(` [ [ "a`, "/0/0"), ""},
		{'"', zeroToken, E(jsonwire.ErrInvalidUTF8).withPos(` [ [ "a`, "/0/0"), ""},
	},
	wantOffset: len(` [ [`),
}, {
	name: jsontest.Name("ErrorPosition/1/0"),
	in:   ` [ "a1" , [ "a` + "\xff" + `2" ] ] `,
	calls: []decoderMethodCall{
		{'[', zeroValue, E(jsonwire.ErrInvalidUTF8).withPos(` [ "a1" , [ "a`, "/1/0"), ""},
		{'[', BeginArray, nil, ""},
		{'"', String("a1"), nil, "/0"},
		{'[', zeroValue, E(jsonwire.ErrInvalidUTF8).withPos(` [ "a1" , [ "a`, "/1/0"), "/0"},
		{'[', BeginArray, nil, "/1"},
		{'"', zeroValue, E(jsonwire.ErrInvalidUTF8).withPos(` [ "a1" , [ "a`, "/1/0"), "/1"},
		{'"', zeroToken, E(jsonwire.ErrInvalidUTF8).withPos(` [ "a1" , [ "a`, "/1/0"), "/1"},
	},
	wantOffset: len(` [ "a1" , [`),
}, {
	name: jsontest.Name("ErrorPosition/0/1"),
	in:   ` [ [ "a2" , "b` + "\xff" + `2" ] ] `,
	calls: []decoderMethodCall{
		{'[', zeroValue, E(jsonwire.ErrInvalidUTF8).withPos(` [ [ "a2" , "b`, "/0/1"), ""},
		{'[', BeginArray, nil, ""},
		{'[', zeroValue, E(jsonwire.ErrInvalidUTF8).withPos(` [ [ "a2" , "b`, "/0/1"), ""},
		{'[', BeginArray, nil, "/0"},
		{'"', String("a2"), nil, "/0/0"},
		{'"', zeroValue, E(jsonwire.ErrInvalidUTF8).withPos(` [ [ "a2" , "b`, "/0/1"), "/0/0"},
		{'"', zeroToken, E(jsonwire.ErrInvalidUTF8).withPos(` [ [ "a2" , "b`, "/0/1"), "/0/0"},
	},
	wantOffset: len(` [ [ "a2"`),
}, {
	name: jsontest.Name("ErrorPosition/1/1"),
	in:   ` [ "a1" , [ "a2" , "b` + "\xff" + `2" ] ] `,
	calls: []decoderMethodCall{
		{'[', zeroValue, E(jsonwire.ErrInvalidUTF8).withPos(` [ "a1" , [ "a2" , "b`, "/1/1"), ""},
		{'[', BeginArray, nil, ""},
		{'"', String("a1"), nil, "/0"},
		{'[', zeroValue, E(jsonwire.ErrInvalidUTF8).withPos(` [ "a1" , [ "a2" , "b`, "/1/1"), ""},
		{'[', BeginArray, nil, "/1"},
		{'"', String("a2"), nil, "/1/0"},
		{'"', zeroValue, E(jsonwire.ErrInvalidUTF8).withPos(` [ "a1" , [ "a2" , "b`, "/1/1"), "/1/0"},
		{'"', zeroToken, E(jsonwire.ErrInvalidUTF8).withPos(` [ "a1" , [ "a2" , "b`, "/1/1"), "/1/0"},
	},
	wantOffset: len(` [ "a1" , [ "a2"`),
}, {
	name: jsontest.Name("ErrorPosition/a1-"),
	in:   ` { "a` + "\xff" + `1" : "b1" } `,
	calls: []decoderMethodCall{
		{'{', zeroValue, E(jsonwire.ErrInvalidUTF8).withPos(` { "a`, ""), ""},
		{'{', BeginObject, nil, ""},
		{'"', zeroValue, E(jsonwire.ErrInvalidUTF8).withPos(` { "a`, ""), ""},
		{'"', zeroToken, E(jsonwire.ErrInvalidUTF8).withPos(` { "a`, ""), ""},
	},
	wantOffset: len(` {`),
}, {
	name: jsontest.Name("ErrorPosition/a1"),
	in:   ` { "a1" : "b` + "\xff" + `1" } `,
	calls: []decoderMethodCall{
		{'{', zeroValue, E(jsonwire.ErrInvalidUTF8).withPos(` { "a1" : "b`, "/a1"), ""},
		{'{', BeginObject, nil, ""},
		{'"', String("a1"), nil, "/a1"},
		{'"', zeroValue, E(jsonwire.ErrInvalidUTF8).withPos(` { "a1" : "b`, "/a1"), ""},
		{'"', zeroToken, E(jsonwire.ErrInvalidUTF8).withPos(` { "a1" : "b`, "/a1"), ""},
	},
	wantOffset: len(` { "a1"`),
}, {
	name: jsontest.Name("ErrorPosition/c1-"),
	in:   ` { "a1" : "b1" , "c` + "\xff" + `1" : "d1" } `,
	calls: []decoderMethodCall{
		{'{', zeroValue, E(jsonwire.ErrInvalidUTF8).withPos(` { "a1" : "b1" , "c`, ""), ""},
		{'{', BeginObject, nil, ""},
		{'"', String("a1"), nil, "/a1"},
		{'"', String("b1"), nil, "/a1"},
		{'"', zeroValue, E(jsonwire.ErrInvalidUTF8).withPos(` { "a1" : "b1" : "c`, ""), "/a1"},
		{'"', zeroToken, E(jsonwire.ErrInvalidUTF8).withPos(` { "a1" : "b1" : "c`, ""), "/a1"},
	},
	wantOffset: len(` { "a1" : "b1"`),
}, {
	name: jsontest.Name("ErrorPosition/c1"),
	in:   ` { "a1" : "b1" , "c1" : "d` + "\xff" + `1" } `,
	calls: []decoderMethodCall{
		{'{', zeroValue, E(jsonwire.ErrInvalidUTF8).withPos(` { "a1" : "b1" , "c1" : "d`, "/c1"), ""},
		{'{', BeginObject, nil, ""},
		{'"', String("a1"), nil, "/a1"},
		{'"', String("b1"), nil, "/a1"},
		{'"', String("c1"), nil, "/c1"},
		{'"', zeroValue, E(jsonwire.ErrInvalidUTF8).withPos(` { "a1" : "b1" : "c1" : "d`, "/c1"), "/c1"},
		{'"', zeroToken, E(jsonwire.ErrInvalidUTF8).withPos(` { "a1" : "b1" : "c1" : "d`, "/c1"), "/c1"},
	},
	wantOffset: len(` { "a1" : "b1" , "c1"`),
}, {
	name: jsontest.Name("ErrorPosition/a1/a2-"),
	in:   ` { "a1" : { "a` + "\xff" + `2" : "b2" } } `,
	calls: []decoderMethodCall{
		{'{', zeroValue, E(jsonwire.ErrInvalidUTF8).withPos(` { "a1" : { "a`, "/a1"), ""},
		{'{', BeginObject, nil, ""},
		{'"', String("a1"), nil, "/a1"},
		{'{', zeroValue, E(jsonwire.ErrInvalidUTF8).withPos(` { "a1" : { "a`, "/a1"), ""},
		{'{', BeginObject, nil, "/a1"},
		{'"', zeroValue, E(jsonwire.ErrInvalidUTF8).withPos(` { "a1" : { "a`, "/a1"), "/a1"},
		{'"', zeroToken, E(jsonwire.ErrInvalidUTF8).withPos(` { "a1" : { "a`, "/a1"), "/a1"},
	},
	wantOffset: len(` { "a1" : {`),
}, {
	name: jsontest.Name("ErrorPosition/a1/a2"),
	in:   ` { "a1" : { "a2" : "b` + "\xff" + `2" } } `,
	calls: []decoderMethodCall{
		{'{', zeroValue, E(jsonwire.ErrInvalidUTF8).withPos(` { "a1" : { "a2" : "b`, "/a1/a2"), ""},
		{'{', BeginObject, nil, ""},
		{'"', String("a1"), nil, "/a1"},
		{'{', zeroValue, E(jsonwire.ErrInvalidUTF8).withPos(` { "a1" : { "a2" : "b`, "/a1/a2"), ""},
		{'{', BeginObject, nil, "/a1"},
		{'"', String("a2"), nil, "/a1/a2"},
		{'"', zeroValue, E(jsonwire.ErrInvalidUTF8).withPos(` { "a1" : { "a2" : "b`, "/a1/a2"), "/a1/a2"},
		{'"', zeroToken, E(jsonwire.ErrInvalidUTF8).withPos(` { "a1" : { "a2" : "b`, "/a1/a2"), "/a1/a2"},
	},
	wantOffset: len(` { "a1" : { "a2"`),
}, {
	name: jsontest.Name("ErrorPosition/a1/c2-"),
	in:   ` { "a1" : { "a2" : "b2" , "c` + "\xff" + `2" : "d2" } } `,
	calls: []decoderMethodCall{
		{'{', zeroValue, E(jsonwire.ErrInvalidUTF8).withPos(` { "a1" : { "a2" : "b2" , "c`, "/a1"), ""},
		{'{', BeginObject, nil, ""},
		{'"', String("a1"), nil, "/a1"},
		{'{', BeginObject, nil, "/a1"},
		{'"', String("a2"), nil, "/a1/a2"},
		{'"', String("b2"), nil, "/a1/a2"},
		{'"', zeroValue, E(jsonwire.ErrInvalidUTF8).withPos(` { "a1" : { "a2" : "b2" , "c`, "/a1"), "/a1/a2"},
		{'"', zeroToken, E(jsonwire.ErrInvalidUTF8).withPos(` { "a1" : { "a2" : "b2" , "c`, "/a1"), "/a1/a2"},
	},
	wantOffset: len(` { "a1" : { "a2" : "b2"`),
}, {
	name: jsontest.Name("ErrorPosition/a1/c2"),
	in:   ` { "a1" : { "a2" : "b2" , "c2" : "d` + "\xff" + `2" } } `,
	calls: []decoderMethodCall{
		{'{', zeroValue, E(jsonwire.ErrInvalidUTF8).withPos(` { "a1" : { "a2" : "b2" , "c2" : "d`, "/a1/c2"), ""},
		{'{', BeginObject, nil, ""},
		{'"', String("a1"), nil, "/a1"},
		{'{', zeroValue, E(jsonwire.ErrInvalidUTF8).withPos(` { "a1" : { "a2" : "b2" , "c2" : "d`, "/a1/c2"), ""},
		{'{', BeginObject, nil, ""},
		{'"', String("a2"), nil, "/a1/a2"},
		{'"', String("b2"), nil, "/a1/a2"},
		{'"', String("c2"), nil, "/a1/c2"},
		{'"', zeroValue, E(jsonwire.ErrInvalidUTF8).withPos(` { "a1" : { "a2" : "b2" , "c2" : "d`, "/a1/c2"), "/a1/c2"},
		{'"', zeroToken, E(jsonwire.ErrInvalidUTF8).withPos(` { "a1" : { "a2" : "b2" , "c2" : "d`, "/a1/c2"), "/a1/c2"},
	},
	wantOffset: len(` { "a1" : { "a2" : "b2" , "c2"`),
}, {
	name: jsontest.Name("ErrorPosition/1/a2"),
	in:   ` [ "a1" , { "a2" : "b` + "\xff" + `2" } ] `,
	calls: []decoderMethodCall{
		{'[', zeroValue, E(jsonwire.ErrInvalidUTF8).withPos(` [ "a1" , { "a2" : "b`, "/1/a2"), ""},
		{'[', BeginArray, nil, ""},
		{'"', String("a1"), nil, "/0"},
		{'{', zeroValue, E(jsonwire.ErrInvalidUTF8).withPos(` [ "a1" , { "a2" : "b`, "/1/a2"), ""},
		{'{', BeginObject, nil, "/1"},
		{'"', String("a2"), nil, "/1/a2"},
		{'"', zeroValue, E(jsonwire.ErrInvalidUTF8).withPos(` [ "a1" , { "a2" : "b`, "/1/a2"), "/1/a2"},
		{'"', zeroToken, E(jsonwire.ErrInvalidUTF8).withPos(` [ "a1" , { "a2" : "b`, "/1/a2"), "/1/a2"},
	},
	wantOffset: len(` [ "a1" , { "a2"`),
}, {
	name: jsontest.Name("ErrorPosition/c1/1"),
	in:   ` { "a1" : "b1" , "c1" : [ "a2" , "b` + "\xff" + `2" ] } `,
	calls: []decoderMethodCall{
		{'{', zeroValue, E(jsonwire.ErrInvalidUTF8).withPos(` { "a1" : "b1" , "c1" : [ "a2" , "b`, "/c1/1"), ""},
		{'{', BeginObject, nil, ""},
		{'"', String("a1"), nil, "/a1"},
		{'"', String("b1"), nil, "/a1"},
		{'"', String("c1"), nil, "/c1"},
		{'[', zeroValue, E(jsonwire.ErrInvalidUTF8).withPos(` { "a1" : "b1" , "c1" : [ "a2" , "b`, "/c1/1"), ""},
		{'[', BeginArray, nil, "/c1"},
		{'"', String("a2"), nil, "/c1/0"},
		{'"', zeroValue, E(jsonwire.ErrInvalidUTF8).withPos(` { "a1" : "b1" , "c1" : [ "a2" , "b`, "/c1/1"), "/c1/0"},
		{'"', zeroToken, E(jsonwire.ErrInvalidUTF8).withPos(` { "a1" : "b1" , "c1" : [ "a2" , "b`, "/c1/1"), "/c1/0"},
	},
	wantOffset: len(` { "a1" : "b1" , "c1" : [ "a2"`),
}, {
	name: jsontest.Name("ErrorPosition/0/a1/1/c3/1"),
	in:   ` [ { "a1" : [ "a2" , { "a3" : "b3" , "c3" : [ "a4" , "b` + "\xff" + `4" ] } ] } ] `,
	calls: []decoderMethodCall{
		{'[', zeroValue, E(jsonwire.ErrInvalidUTF8).withPos(` [ { "a1" : [ "a2" , { "a3" : "b3" , "c3" : [ "a4" , "b`, "/0/a1/1/c3/1"), ""},
		{'[', BeginArray, nil, ""},
		{'{', zeroValue, E(jsonwire.ErrInvalidUTF8).withPos(` [ { "a1" : [ "a2" , { "a3" : "b3" , "c3" : [ "a4" , "b`, "/0/a1/1/c3/1"), ""},
		{'{', BeginObject, nil, "/0"},
		{'"', String("a1"), nil, "/0/a1"},
		{'[', zeroValue, E(jsonwire.ErrInvalidUTF8).withPos(` [ { "a1" : [ "a2" , { "a3" : "b3" , "c3" : [ "a4" , "b`, "/0/a1/1/c3/1"), ""},
		{'[', BeginArray, nil, ""},
		{'"', String("a2"), nil, "/0/a1/0"},
		{'{', zeroValue, E(jsonwire.ErrInvalidUTF8).withPos(` [ { "a1" : [ "a2" , { "a3" : "b3" , "c3" : [ "a4" , "b`, "/0/a1/1/c3/1"), ""},
		{'{', BeginObject, nil, "/0/a1/1"},
		{'"', String("a3"), nil, "/0/a1/1/a3"},
		{'"', String("b3"), nil, "/0/a1/1/a3"},
		{'"', String("c3"), nil, "/0/a1/1/c3"},
		{'[', zeroValue, E(jsonwire.ErrInvalidUTF8).withPos(` [ { "a1" : [ "a2" , { "a3" : "b3" , "c3" : [ "a4" , "b`, "/0/a1/1/c3/1"), ""},
		{'[', BeginArray, nil, "/0/a1/1/c3"},
		{'"', String("a4"), nil, "/0/a1/1/c3/0"},
		{'"', zeroValue, E(jsonwire.ErrInvalidUTF8).withPos(` [ { "a1" : [ "a2" , { "a3" : "b3" , "c3" : [ "a4" , "b`, "/0/a1/1/c3/1"), "/0/a1/1/c3/0"},
		{'"', zeroToken, E(jsonwire.ErrInvalidUTF8).withPos(` [ { "a1" : [ "a2" , { "a3" : "b3" , "c3" : [ "a4" , "b`, "/0/a1/1/c3/1"), "/0/a1/1/c3/0"},
	},
	wantOffset: len(` [ { "a1" : [ "a2" , { "a3" : "b3" , "c3" : [ "a4"`),
}}

// TestDecoderErrors test that Decoder errors occur when we expect and
// leaves the Decoder in a consistent state.
func TestDecoderErrors(t *testing.T) {
	for _, td := range decoderErrorTestdata {
		t.Run(path.Join(td.name.Name), func(t *testing.T) {
			testDecoderErrors(t, td.name.Where, td.opts, td.in, td.calls, td.wantOffset)
		})
	}
}
func testDecoderErrors(t *testing.T, where jsontest.CasePos, opts []Options, in string, calls []decoderMethodCall, wantOffset int) {
	src := bytes.NewBufferString(in)
	dec := NewDecoder(src, opts...)
	for i, call := range calls {
		gotKind := dec.PeekKind()
		if gotKind != call.wantKind {
			t.Fatalf("%s: %d: Decoder.PeekKind = %v, want %v", where, i, gotKind, call.wantKind)
		}

		var gotErr error
		switch wantOut := call.wantOut.(type) {
		case Token:
			var gotOut Token
			gotOut, gotErr = dec.ReadToken()
			if gotOut.String() != wantOut.String() {
				t.Fatalf("%s: %d: Decoder.ReadToken = %v, want %v", where, i, gotOut, wantOut)
			}
		case Value:
			var gotOut Value
			gotOut, gotErr = dec.ReadValue()
			if string(gotOut) != string(wantOut) {
				t.Fatalf("%s: %d: Decoder.ReadValue = %s, want %s", where, i, gotOut, wantOut)
			}
		}
		if !equalError(gotErr, call.wantErr) {
			t.Fatalf("%s: %d: error mismatch:\ngot  %v\nwant %v", where, i, gotErr, call.wantErr)
		}
		if call.wantPointer != "" {
			gotPointer := dec.StackPointer()
			if gotPointer != call.wantPointer {
				t.Fatalf("%s: %d: Decoder.StackPointer = %s, want %s", where, i, gotPointer, call.wantPointer)
			}
		}
	}
	gotOffset := int(dec.InputOffset())
	if gotOffset != wantOffset {
		t.Fatalf("%s: Decoder.InputOffset = %v, want %v", where, gotOffset, wantOffset)
	}
	gotUnread := string(dec.s.unreadBuffer()) // should be a prefix of wantUnread
	wantUnread := in[wantOffset:]
	if !strings.HasPrefix(wantUnread, gotUnread) {
		t.Fatalf("%s: Decoder.UnreadBuffer = %v, want %v", where, gotUnread, wantUnread)
	}
}

// TestBufferDecoder tests that we detect misuses of bytes.Buffer with Decoder.
func TestBufferDecoder(t *testing.T) {
	bb := bytes.NewBufferString("[null, false, true]")
	dec := NewDecoder(bb)
	var err error
	for {
		if _, err = dec.ReadToken(); err != nil {
			break
		}
		bb.WriteByte(' ') // not allowed to write to the buffer while reading
	}
	want := &ioError{action: "read", err: errBufferWriteAfterNext}
	if !equalError(err, want) {
		t.Fatalf("error mismatch: got %v, want %v", err, want)
	}
}

var resumableDecoderTestdata = []string{
	`0`,
	`123456789`,
	`0.0`,
	`0.123456789`,
	`0e0`,
	`0e+0`,
	`0e123456789`,
	`0e+123456789`,
	`123456789.123456789e+123456789`,
	`-0`,
	`-123456789`,
	`-0.0`,
	`-0.123456789`,
	`-0e0`,
	`-0e-0`,
	`-0e123456789`,
	`-0e-123456789`,
	`-123456789.123456789e-123456789`,

	`""`,
	`"a"`,
	`"ab"`,
	`"abc"`,
	`"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"`,
	`"\"\\\/\b\f\n\r\t"`,
	`"\u0022\u005c\u002f\u0008\u000c\u000a\u000d\u0009"`,
	`"\ud800\udead"`,
	"\"\u0080\u00f6\u20ac\ud799\ue000\ufb33\ufffd\U0001f602\"",
	`"\u0080\u00f6\u20ac\ud799\ue000\ufb33\ufffd\ud83d\ude02"`,
}

// TestResumableDecoder tests that resume logic for parsing a
// JSON string and number properly works across every possible split point.
func TestResumableDecoder(t *testing.T) {
	for _, want := range resumableDecoderTestdata {
		t.Run("", func(t *testing.T) {
			dec := NewDecoder(iotest.OneByteReader(strings.NewReader(want)))
			got, err := dec.ReadValue()
			if err != nil {
				t.Fatalf("Decoder.ReadValue error: %v", err)
			}
			if string(got) != want {
				t.Fatalf("Decoder.ReadValue = %s, want %s", got, want)
			}
		})
	}
}

// TestBlockingDecoder verifies that JSON values except numbers can be
// synchronously sent and received on a blocking pipe without a deadlock.
// Numbers are the exception since termination cannot be determined until
// either the pipe ends or a non-numeric character is encountered.
func TestBlockingDecoder(t *testing.T) {
	values := []string{"null", "false", "true", `""`, `{}`, `[]`}

	r, w := net.Pipe()
	defer r.Close()
	defer w.Close()

	enc := NewEncoder(w, jsonflags.OmitTopLevelNewline|1)
	dec := NewDecoder(r)

	errCh := make(chan error)

	// Test synchronous ReadToken calls.
	for _, want := range values {
		go func() {
			errCh <- enc.WriteValue(Value(want))
		}()

		tok, err := dec.ReadToken()
		if err != nil {
			t.Fatalf("Decoder.ReadToken error: %v", err)
		}
		got := tok.String()
		switch tok.Kind() {
		case '"':
			got = `"` + got + `"`
		case '{', '[':
			tok, err := dec.ReadToken()
			if err != nil {
				t.Fatalf("Decoder.ReadToken error: %v", err)
			}
			got += tok.String()
		}
		if got != want {
			t.Fatalf("ReadTokens = %s, want %s", got, want)
		}

		if err := <-errCh; err != nil {
			t.Fatalf("Encoder.WriteValue error: %v", err)
		}
	}

	// Test synchronous ReadValue calls.
	for _, want := range values {
		go func() {
			errCh <- enc.WriteValue(Value(want))
		}()

		got, err := dec.ReadValue()
		if err != nil {
			t.Fatalf("Decoder.ReadValue error: %v", err)
		}
		if string(got) != want {
			t.Fatalf("ReadValue = %s, want %s", got, want)
		}

		if err := <-errCh; err != nil {
			t.Fatalf("Encoder.WriteValue error: %v", err)
		}
	}
}

func TestPeekableDecoder(t *testing.T) {
	type operation any // PeekKind | ReadToken | ReadValue | BufferWrite
	type PeekKind struct {
		want Kind
	}
	type ReadToken struct {
		wantKind Kind
		wantErr  error
	}
	type ReadValue struct {
		wantKind Kind
		wantErr  error
	}
	type WriteString struct {
		in string
	}
	ops := []operation{
		PeekKind{0},
		WriteString{"[ "},
		ReadToken{0, io.EOF}, // previous error from PeekKind is cached once
		ReadToken{'[', nil},

		PeekKind{0},
		WriteString{"] "},
		ReadValue{0, E(io.ErrUnexpectedEOF).withPos("[ ", "")}, // previous error from PeekKind is cached once
		ReadValue{0, newInvalidCharacterError("]", "at start of value").withPos("[ ", "/0")},
		ReadToken{']', nil},

		WriteString{"[ "},
		ReadToken{'[', nil},

		WriteString{" null "},
		PeekKind{'n'},
		PeekKind{'n'},
		ReadToken{'n', nil},

		WriteString{", "},
		PeekKind{0},
		WriteString{"fal"},
		PeekKind{'f'},
		ReadValue{0, E(io.ErrUnexpectedEOF).withPos("[ ] [  null , fal", "/1")},
		WriteString{"se "},
		ReadValue{'f', nil},

		PeekKind{0},
		WriteString{" , "},
		PeekKind{0},
		WriteString{` "" `},
		ReadValue{0, E(io.ErrUnexpectedEOF).withPos("[ ] [  null , false  , ", "")}, // previous error from PeekKind is cached once
		ReadValue{'"', nil},

		WriteString{" , 0"},
		PeekKind{'0'},
		ReadToken{'0', nil},

		WriteString{" , {} , []"},
		PeekKind{'{'},
		ReadValue{'{', nil},
		ReadValue{'[', nil},

		WriteString{"]"},
		ReadToken{']', nil},
	}

	bb := struct{ *bytes.Buffer }{new(bytes.Buffer)}
	d := NewDecoder(bb)
	for i, op := range ops {
		switch op := op.(type) {
		case PeekKind:
			if got := d.PeekKind(); got != op.want {
				t.Fatalf("%d: Decoder.PeekKind() = %v, want %v", i, got, op.want)
			}
		case ReadToken:
			gotTok, gotErr := d.ReadToken()
			gotKind := gotTok.Kind()
			if gotKind != op.wantKind || !equalError(gotErr, op.wantErr) {
				t.Fatalf("%d: Decoder.ReadToken() = (%v, %v), want (%v, %v)", i, gotKind, gotErr, op.wantKind, op.wantErr)
			}
		case ReadValue:
			gotVal, gotErr := d.ReadValue()
			gotKind := gotVal.Kind()
			if gotKind != op.wantKind || !equalError(gotErr, op.wantErr) {
				t.Fatalf("%d: Decoder.ReadValue() = (%v, %v), want (%v, %v)", i, gotKind, gotErr, op.wantKind, op.wantErr)
			}
		case WriteString:
			bb.WriteString(op.in)
		default:
			panic(fmt.Sprintf("unknown operation: %T", op))
		}
	}
}

// TestDecoderReset tests that the decoder preserves its internal
// buffer between Reset calls to avoid frequent allocations when reusing the decoder.
// It ensures that the buffer capacity is maintained while avoiding aliasing
// issues with [bytes.Buffer].
func TestDecoderReset(t *testing.T) {
	// Create a decoder with a reasonably large JSON input to ensure buffer growth.
	largeJSON := `{"key1":"value1","key2":"value2","key3":"value3","key4":"value4","key5":"value5"}`
	dec := NewDecoder(strings.NewReader(largeJSON))

	t.Run("Test capacity preservation", func(t *testing.T) {
		// Read the first JSON value to grow the internal buffer.
		val1, err := dec.ReadValue()
		if err != nil {
			t.Fatalf("first ReadValue failed: %v", err)
		}
		if string(val1) != largeJSON {
			t.Fatalf("first ReadValue = %q, want %q", val1, largeJSON)
		}

		// Get the buffer capacity after first use.
		initialCapacity := cap(dec.s.buf)
		if initialCapacity == 0 {
			t.Fatalf("expected non-zero buffer capacity after first use")
		}

		// Reset with a new reader - this should preserve the buffer capacity.
		dec.Reset(strings.NewReader(largeJSON))

		// Verify the buffer capacity is preserved (or at least not smaller).
		preservedCapacity := cap(dec.s.buf)
		if preservedCapacity < initialCapacity {
			t.Fatalf("buffer capacity reduced after Reset: got %d, want at least %d", preservedCapacity, initialCapacity)
		}

		// Read the second JSON value to ensure the decoder still works correctly.
		val2, err := dec.ReadValue()
		if err != nil {
			t.Fatalf("second ReadValue failed: %v", err)
		}
		if string(val2) != largeJSON {
			t.Fatalf("second ReadValue = %q, want %q", val2, largeJSON)
		}
	})

	var bbBuf []byte
	t.Run("Test aliasing with bytes.Buffer", func(t *testing.T) {
		// Test with bytes.Buffer to verify proper aliasing behavior.
		bb := bytes.NewBufferString(largeJSON)
		dec.Reset(bb)
		bbBuf = bb.Bytes()

		// Read the third JSON value to ensure functionality with bytes.Buffer.
		val3, err := dec.ReadValue()
		if err != nil {
			t.Fatalf("fourth ReadValue failed: %v", err)
		}
		if string(val3) != largeJSON {
			t.Fatalf("fourth ReadValue = %q, want %q", val3, largeJSON)
		}
		// The decoder buffer should alias bytes.Buffer's internal buffer.
		if len(dec.s.buf) == 0 || len(bbBuf) == 0 || &dec.s.buf[0] != &bbBuf[0] {
			t.Fatalf("decoder buffer does not alias bytes.Buffer")
		}
	})

	t.Run("Test aliasing removed after Reset", func(t *testing.T) {
		// Reset with a new reader and verify the buffer is not aliased.
		dec.Reset(strings.NewReader(largeJSON))
		val4, err := dec.ReadValue()
		if err != nil {
			t.Fatalf("fifth ReadValue failed: %v", err)
		}
		if string(val4) != largeJSON {
			t.Fatalf("fourth ReadValue = %q, want %q", val4, largeJSON)
		}

		// The decoder buffer should not alias the bytes.Buffer's internal buffer.
		if len(dec.s.buf) == 0 || len(bbBuf) == 0 || &dec.s.buf[0] == &bbBuf[0] {
			t.Fatalf("decoder buffer aliases bytes.Buffer")
		}
	})
}
