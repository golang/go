// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jsonv2

package jsontext

import (
	"bytes"
	"errors"
	"io"
	"path"
	"slices"
	"testing"

	"encoding/json/internal/jsonflags"
	"encoding/json/internal/jsontest"
	"encoding/json/internal/jsonwire"
)

// TestEncoder tests whether we can produce JSON with either tokens or raw values.
func TestEncoder(t *testing.T) {
	for _, td := range coderTestdata {
		for _, formatName := range []string{"Compact", "Indented"} {
			for _, typeName := range []string{"Token", "Value", "TokenDelims"} {
				t.Run(path.Join(td.name.Name, typeName, formatName), func(t *testing.T) {
					testEncoder(t, td.name.Where, formatName, typeName, td)
				})
			}
		}
	}
}
func testEncoder(t *testing.T, where jsontest.CasePos, formatName, typeName string, td coderTestdataEntry) {
	var want string
	var opts []Options
	dst := new(bytes.Buffer)
	opts = append(opts, jsonflags.OmitTopLevelNewline|1)
	want = td.outCompacted
	switch formatName {
	case "Indented":
		opts = append(opts, Multiline(true))
		opts = append(opts, WithIndentPrefix("\t"))
		opts = append(opts, WithIndent("    "))
		if td.outIndented != "" {
			want = td.outIndented
		}
	}
	enc := NewEncoder(dst, opts...)

	switch typeName {
	case "Token":
		var pointers []Pointer
		for _, tok := range td.tokens {
			if err := enc.WriteToken(tok); err != nil {
				t.Fatalf("%s: Encoder.WriteToken error: %v", where, err)
			}
			if td.pointers != nil {
				pointers = append(pointers, enc.StackPointer())
			}
		}
		if !slices.Equal(pointers, td.pointers) {
			t.Fatalf("%s: pointers mismatch:\ngot  %q\nwant %q", where, pointers, td.pointers)
		}
	case "Value":
		if err := enc.WriteValue(Value(td.in)); err != nil {
			t.Fatalf("%s: Encoder.WriteValue error: %v", where, err)
		}
	case "TokenDelims":
		// Use WriteToken for object/array delimiters, WriteValue otherwise.
		for _, tok := range td.tokens {
			switch tok.Kind() {
			case '{', '}', '[', ']':
				if err := enc.WriteToken(tok); err != nil {
					t.Fatalf("%s: Encoder.WriteToken error: %v", where, err)
				}
			default:
				val := Value(tok.String())
				if tok.Kind() == '"' {
					val, _ = jsonwire.AppendQuote(nil, tok.String(), &jsonflags.Flags{})
				}
				if err := enc.WriteValue(val); err != nil {
					t.Fatalf("%s: Encoder.WriteValue error: %v", where, err)
				}
			}
		}
	}

	got := dst.String()
	if got != want {
		t.Errorf("%s: output mismatch:\ngot  %q\nwant %q", where, got, want)
	}
}

// TestFaultyEncoder tests that temporary I/O errors are not fatal.
func TestFaultyEncoder(t *testing.T) {
	for _, td := range coderTestdata {
		for _, typeName := range []string{"Token", "Value"} {
			t.Run(path.Join(td.name.Name, typeName), func(t *testing.T) {
				testFaultyEncoder(t, td.name.Where, typeName, td)
			})
		}
	}
}
func testFaultyEncoder(t *testing.T, where jsontest.CasePos, typeName string, td coderTestdataEntry) {
	b := &FaultyBuffer{
		MaxBytes: 1,
		MayError: io.ErrShortWrite,
	}

	// Write all the tokens.
	// Even if the underlying io.Writer may be faulty,
	// writing a valid token or value is guaranteed to at least
	// be appended to the internal buffer.
	// In other words, syntactic errors occur before I/O errors.
	enc := NewEncoder(b)
	switch typeName {
	case "Token":
		for i, tok := range td.tokens {
			err := enc.WriteToken(tok)
			if err != nil && !errors.Is(err, io.ErrShortWrite) {
				t.Fatalf("%s: %d: Encoder.WriteToken error: %v", where, i, err)
			}
		}
	case "Value":
		err := enc.WriteValue(Value(td.in))
		if err != nil && !errors.Is(err, io.ErrShortWrite) {
			t.Fatalf("%s: Encoder.WriteValue error: %v", where, err)
		}
	}
	gotOutput := string(append(b.B, enc.s.unflushedBuffer()...))
	wantOutput := td.outCompacted + "\n"
	if gotOutput != wantOutput {
		t.Fatalf("%s: output mismatch:\ngot  %s\nwant %s", where, gotOutput, wantOutput)
	}
}

type encoderMethodCall struct {
	in          tokOrVal
	wantErr     error
	wantPointer Pointer
}

var encoderErrorTestdata = []struct {
	name    jsontest.CaseName
	opts    []Options
	calls   []encoderMethodCall
	wantOut string
}{{
	name: jsontest.Name("InvalidToken"),
	calls: []encoderMethodCall{
		{zeroToken, E(errInvalidToken), ""},
	},
}, {
	name: jsontest.Name("InvalidValue"),
	calls: []encoderMethodCall{
		{Value(`#`), newInvalidCharacterError("#", "at start of value"), ""},
	},
}, {
	name: jsontest.Name("InvalidValue/DoubleZero"),
	calls: []encoderMethodCall{
		{Value(`00`), newInvalidCharacterError("0", "after top-level value").withPos(`0`, ""), ""},
	},
}, {
	name: jsontest.Name("TruncatedValue"),
	calls: []encoderMethodCall{
		{zeroValue, E(io.ErrUnexpectedEOF).withPos("", ""), ""},
	},
}, {
	name: jsontest.Name("TruncatedNull"),
	calls: []encoderMethodCall{
		{Value(`nul`), E(io.ErrUnexpectedEOF).withPos("nul", ""), ""},
	},
}, {
	name: jsontest.Name("InvalidNull"),
	calls: []encoderMethodCall{
		{Value(`nulL`), newInvalidCharacterError("L", "in literal null (expecting 'l')").withPos(`nul`, ""), ""},
	},
}, {
	name: jsontest.Name("TruncatedFalse"),
	calls: []encoderMethodCall{
		{Value(`fals`), E(io.ErrUnexpectedEOF).withPos("fals", ""), ""},
	},
}, {
	name: jsontest.Name("InvalidFalse"),
	calls: []encoderMethodCall{
		{Value(`falsE`), newInvalidCharacterError("E", "in literal false (expecting 'e')").withPos(`fals`, ""), ""},
	},
}, {
	name: jsontest.Name("TruncatedTrue"),
	calls: []encoderMethodCall{
		{Value(`tru`), E(io.ErrUnexpectedEOF).withPos(`tru`, ""), ""},
	},
}, {
	name: jsontest.Name("InvalidTrue"),
	calls: []encoderMethodCall{
		{Value(`truE`), newInvalidCharacterError("E", "in literal true (expecting 'e')").withPos(`tru`, ""), ""},
	},
}, {
	name: jsontest.Name("TruncatedString"),
	calls: []encoderMethodCall{
		{Value(`"star`), E(io.ErrUnexpectedEOF).withPos(`"star`, ""), ""},
	},
}, {
	name: jsontest.Name("InvalidString"),
	calls: []encoderMethodCall{
		{Value(`"ok` + "\x00"), newInvalidCharacterError("\x00", `in string (expecting non-control character)`).withPos(`"ok`, ""), ""},
	},
}, {
	name: jsontest.Name("ValidString/AllowInvalidUTF8/Token"),
	opts: []Options{AllowInvalidUTF8(true)},
	calls: []encoderMethodCall{
		{String("living\xde\xad\xbe\xef"), nil, ""},
	},
	wantOut: "\"living\xde\xad\ufffd\ufffd\"\n",
}, {
	name: jsontest.Name("ValidString/AllowInvalidUTF8/Value"),
	opts: []Options{AllowInvalidUTF8(true)},
	calls: []encoderMethodCall{
		{Value("\"living\xde\xad\xbe\xef\""), nil, ""},
	},
	wantOut: "\"living\xde\xad\ufffd\ufffd\"\n",
}, {
	name: jsontest.Name("InvalidString/RejectInvalidUTF8"),
	opts: []Options{AllowInvalidUTF8(false)},
	calls: []encoderMethodCall{
		{String("living\xde\xad\xbe\xef"), E(jsonwire.ErrInvalidUTF8), ""},
		{Value("\"living\xde\xad\xbe\xef\""), E(jsonwire.ErrInvalidUTF8).withPos("\"living\xde\xad", ""), ""},
		{BeginObject, nil, ""},
		{String("name"), nil, ""},
		{BeginArray, nil, ""},
		{String("living\xde\xad\xbe\xef"), E(jsonwire.ErrInvalidUTF8).withPos(`{"name":[`, "/name/0"), ""},
		{Value("\"living\xde\xad\xbe\xef\""), E(jsonwire.ErrInvalidUTF8).withPos("{\"name\":[\"living\xde\xad", "/name/0"), ""},
	},
	wantOut: `{"name":[`,
}, {
	name: jsontest.Name("TruncatedNumber"),
	calls: []encoderMethodCall{
		{Value(`0.`), E(io.ErrUnexpectedEOF).withPos("0", ""), ""},
	},
}, {
	name: jsontest.Name("InvalidNumber"),
	calls: []encoderMethodCall{
		{Value(`0.e`), newInvalidCharacterError("e", "in number (expecting digit)").withPos(`0.`, ""), ""},
	},
}, {
	name: jsontest.Name("TruncatedObject/AfterStart"),
	calls: []encoderMethodCall{
		{Value(`{`), E(io.ErrUnexpectedEOF).withPos("{", ""), ""},
	},
}, {
	name: jsontest.Name("TruncatedObject/AfterName"),
	calls: []encoderMethodCall{
		{Value(`{"X"`), E(io.ErrUnexpectedEOF).withPos(`{"X"`, "/X"), ""},
	},
}, {
	name: jsontest.Name("TruncatedObject/AfterColon"),
	calls: []encoderMethodCall{
		{Value(`{"X":`), E(io.ErrUnexpectedEOF).withPos(`{"X":`, "/X"), ""},
	},
}, {
	name: jsontest.Name("TruncatedObject/AfterValue"),
	calls: []encoderMethodCall{
		{Value(`{"0":0`), E(io.ErrUnexpectedEOF).withPos(`{"0":0`, ""), ""},
	},
}, {
	name: jsontest.Name("TruncatedObject/AfterComma"),
	calls: []encoderMethodCall{
		{Value(`{"0":0,`), E(io.ErrUnexpectedEOF).withPos(`{"0":0,`, ""), ""},
	},
}, {
	name: jsontest.Name("InvalidObject/MissingColon"),
	calls: []encoderMethodCall{
		{Value(` { "fizz" "buzz" } `), newInvalidCharacterError("\"", "after object name (expecting ':')").withPos(` { "fizz" `, "/fizz"), ""},
		{Value(` { "fizz" , "buzz" } `), newInvalidCharacterError(",", "after object name (expecting ':')").withPos(` { "fizz" `, "/fizz"), ""},
	},
}, {
	name: jsontest.Name("InvalidObject/MissingComma"),
	calls: []encoderMethodCall{
		{Value(` { "fizz" : "buzz" "gazz" } `), newInvalidCharacterError("\"", "after object value (expecting ',' or '}')").withPos(` { "fizz" : "buzz" `, ""), ""},
		{Value(` { "fizz" : "buzz" : "gazz" } `), newInvalidCharacterError(":", "after object value (expecting ',' or '}')").withPos(` { "fizz" : "buzz" `, ""), ""},
	},
}, {
	name: jsontest.Name("InvalidObject/ExtraComma"),
	calls: []encoderMethodCall{
		{Value(` { , } `), newInvalidCharacterError(",", `at start of string (expecting '"')`).withPos(` { `, ""), ""},
		{Value(` { "fizz" : "buzz" , } `), newInvalidCharacterError("}", `at start of string (expecting '"')`).withPos(` { "fizz" : "buzz" , `, ""), ""},
	},
}, {
	name: jsontest.Name("InvalidObject/InvalidName"),
	calls: []encoderMethodCall{
		{Value(`{ null }`), newInvalidCharacterError("n", `at start of string (expecting '"')`).withPos(`{ `, ""), ""},
		{Value(`{ false }`), newInvalidCharacterError("f", `at start of string (expecting '"')`).withPos(`{ `, ""), ""},
		{Value(`{ true }`), newInvalidCharacterError("t", `at start of string (expecting '"')`).withPos(`{ `, ""), ""},
		{Value(`{ 0 }`), newInvalidCharacterError("0", `at start of string (expecting '"')`).withPos(`{ `, ""), ""},
		{Value(`{ {} }`), newInvalidCharacterError("{", `at start of string (expecting '"')`).withPos(`{ `, ""), ""},
		{Value(`{ [] }`), newInvalidCharacterError("[", `at start of string (expecting '"')`).withPos(`{ `, ""), ""},
		{BeginObject, nil, ""},
		{Null, E(ErrNonStringName).withPos(`{`, ""), ""},
		{Value(`null`), E(ErrNonStringName).withPos(`{`, ""), ""},
		{False, E(ErrNonStringName).withPos(`{`, ""), ""},
		{Value(`false`), E(ErrNonStringName).withPos(`{`, ""), ""},
		{True, E(ErrNonStringName).withPos(`{`, ""), ""},
		{Value(`true`), E(ErrNonStringName).withPos(`{`, ""), ""},
		{Uint(0), E(ErrNonStringName).withPos(`{`, ""), ""},
		{Value(`0`), E(ErrNonStringName).withPos(`{`, ""), ""},
		{BeginObject, E(ErrNonStringName).withPos(`{`, ""), ""},
		{Value(`{}`), E(ErrNonStringName).withPos(`{`, ""), ""},
		{BeginArray, E(ErrNonStringName).withPos(`{`, ""), ""},
		{Value(`[]`), E(ErrNonStringName).withPos(`{`, ""), ""},
		{EndObject, nil, ""},
	},
	wantOut: "{}\n",
}, {
	name: jsontest.Name("InvalidObject/InvalidValue"),
	calls: []encoderMethodCall{
		{Value(`{ "0": x }`), newInvalidCharacterError("x", `at start of value`).withPos(`{ "0": `, "/0"), ""},
	},
}, {
	name: jsontest.Name("InvalidObject/MismatchingDelim"),
	calls: []encoderMethodCall{
		{Value(` { ] `), newInvalidCharacterError("]", `at start of string (expecting '"')`).withPos(` { `, ""), ""},
		{Value(` { "0":0 ] `), newInvalidCharacterError("]", `after object value (expecting ',' or '}')`).withPos(` { "0":0 `, ""), ""},
		{BeginObject, nil, ""},
		{EndArray, E(errMismatchDelim).withPos(`{`, ""), ""},
		{Value(`]`), newInvalidCharacterError("]", "at start of value").withPos(`{`, ""), ""},
		{EndObject, nil, ""},
	},
	wantOut: "{}\n",
}, {
	name: jsontest.Name("ValidObject/UniqueNames"),
	calls: []encoderMethodCall{
		{BeginObject, nil, ""},
		{String("0"), nil, ""},
		{Uint(0), nil, ""},
		{String("1"), nil, ""},
		{Uint(1), nil, ""},
		{EndObject, nil, ""},
		{Value(` { "0" : 0 , "1" : 1 } `), nil, ""},
	},
	wantOut: `{"0":0,"1":1}` + "\n" + `{"0":0,"1":1}` + "\n",
}, {
	name: jsontest.Name("ValidObject/DuplicateNames"),
	opts: []Options{AllowDuplicateNames(true)},
	calls: []encoderMethodCall{
		{BeginObject, nil, ""},
		{String("0"), nil, ""},
		{Uint(0), nil, ""},
		{String("0"), nil, ""},
		{Uint(0), nil, ""},
		{EndObject, nil, ""},
		{Value(` { "0" : 0 , "0" : 0 } `), nil, ""},
	},
	wantOut: `{"0":0,"0":0}` + "\n" + `{"0":0,"0":0}` + "\n",
}, {
	name: jsontest.Name("InvalidObject/DuplicateNames"),
	calls: []encoderMethodCall{
		{BeginObject, nil, ""},
		{String("X"), nil, ""},
		{BeginObject, nil, ""},
		{EndObject, nil, ""},
		{String("X"), E(ErrDuplicateName).withPos(`{"X":{},`, "/X"), "/X"},
		{Value(`"X"`), E(ErrDuplicateName).withPos(`{"X":{},`, "/X"), "/X"},
		{String("Y"), nil, ""},
		{BeginObject, nil, ""},
		{EndObject, nil, ""},
		{String("X"), E(ErrDuplicateName).withPos(`{"X":{},"Y":{},`, "/X"), "/Y"},
		{Value(`"X"`), E(ErrDuplicateName).withPos(`{"X":{},"Y":{},`, "/X"), "/Y"},
		{String("Y"), E(ErrDuplicateName).withPos(`{"X":{},"Y":{},`, "/Y"), "/Y"},
		{Value(`"Y"`), E(ErrDuplicateName).withPos(`{"X":{},"Y":{},`, "/Y"), "/Y"},
		{EndObject, nil, ""},
		{Value(` { "X" : 0 , "Y" : 1 , "X" : 0 } `), E(ErrDuplicateName).withPos(`{"X":{},"Y":{}}`+"\n"+` { "X" : 0 , "Y" : 1 , `, "/X"), ""},
	},
	wantOut: `{"X":{},"Y":{}}` + "\n",
}, {
	name: jsontest.Name("TruncatedArray/AfterStart"),
	calls: []encoderMethodCall{
		{Value(`[`), E(io.ErrUnexpectedEOF).withPos(`[`, ""), ""},
	},
}, {
	name: jsontest.Name("TruncatedArray/AfterValue"),
	calls: []encoderMethodCall{
		{Value(`[0`), E(io.ErrUnexpectedEOF).withPos(`[0`, ""), ""},
	},
}, {
	name: jsontest.Name("TruncatedArray/AfterComma"),
	calls: []encoderMethodCall{
		{Value(`[0,`), E(io.ErrUnexpectedEOF).withPos(`[0,`, ""), ""},
	},
}, {
	name: jsontest.Name("TruncatedArray/MissingComma"),
	calls: []encoderMethodCall{
		{Value(` [ "fizz" "buzz" ] `), newInvalidCharacterError("\"", "after array value (expecting ',' or ']')").withPos(` [ "fizz" `, ""), ""},
	},
}, {
	name: jsontest.Name("InvalidArray/MismatchingDelim"),
	calls: []encoderMethodCall{
		{Value(` [ } `), newInvalidCharacterError("}", `at start of value`).withPos(` [ `, "/0"), ""},
		{BeginArray, nil, ""},
		{EndObject, E(errMismatchDelim).withPos(`[`, "/0"), ""},
		{Value(`}`), newInvalidCharacterError("}", "at start of value").withPos(`[`, "/0"), ""},
		{EndArray, nil, ""},
	},
	wantOut: "[]\n",
}, {
	name:    jsontest.Name("Format/Object/SpaceAfterColon"),
	opts:    []Options{SpaceAfterColon(true)},
	calls:   []encoderMethodCall{{Value(`{"fizz":"buzz","wizz":"wuzz"}`), nil, ""}},
	wantOut: "{\"fizz\": \"buzz\",\"wizz\": \"wuzz\"}\n",
}, {
	name:    jsontest.Name("Format/Object/SpaceAfterComma"),
	opts:    []Options{SpaceAfterComma(true)},
	calls:   []encoderMethodCall{{Value(`{"fizz":"buzz","wizz":"wuzz"}`), nil, ""}},
	wantOut: "{\"fizz\":\"buzz\", \"wizz\":\"wuzz\"}\n",
}, {
	name:    jsontest.Name("Format/Object/SpaceAfterColonAndComma"),
	opts:    []Options{SpaceAfterColon(true), SpaceAfterComma(true)},
	calls:   []encoderMethodCall{{Value(`{"fizz":"buzz","wizz":"wuzz"}`), nil, ""}},
	wantOut: "{\"fizz\": \"buzz\", \"wizz\": \"wuzz\"}\n",
}, {
	name:    jsontest.Name("Format/Object/NoSpaceAfterColon+SpaceAfterComma+Multiline"),
	opts:    []Options{SpaceAfterColon(false), SpaceAfterComma(true), Multiline(true)},
	calls:   []encoderMethodCall{{Value(`{"fizz":"buzz","wizz":"wuzz"}`), nil, ""}},
	wantOut: "{\n\t\"fizz\":\"buzz\", \n\t\"wizz\":\"wuzz\"\n}\n",
}, {
	name:    jsontest.Name("Format/Array/SpaceAfterComma"),
	opts:    []Options{SpaceAfterComma(true)},
	calls:   []encoderMethodCall{{Value(`["fizz","buzz"]`), nil, ""}},
	wantOut: "[\"fizz\", \"buzz\"]\n",
}, {
	name:    jsontest.Name("Format/Array/NoSpaceAfterComma+Multiline"),
	opts:    []Options{SpaceAfterComma(false), Multiline(true)},
	calls:   []encoderMethodCall{{Value(`["fizz","buzz"]`), nil, ""}},
	wantOut: "[\n\t\"fizz\",\n\t\"buzz\"\n]\n",
}, {
	name: jsontest.Name("Format/ReorderWithWhitespace"),
	opts: []Options{
		AllowDuplicateNames(true),
		AllowInvalidUTF8(true),
		ReorderRawObjects(true),
		SpaceAfterComma(true),
		SpaceAfterColon(false),
		Multiline(true),
		WithIndentPrefix("    "),
		WithIndent("\t"),
		PreserveRawStrings(true),
	},
	calls: []encoderMethodCall{
		{BeginArray, nil, ""},
		{BeginArray, nil, ""},
		{Value(` { "fizz" : "buzz" ,
			"zip" : {
				"x` + "\xfd" + `x" : 123 , "x` + "\xff" + `x" : 123, "x` + "\xfe" + `x" : 123
			},
			"zap" : {
				"xxx" : 333, "xxx": 1, "xxx": 22
			},
			"alpha" : "bravo" } `), nil, ""},
		{EndArray, nil, ""},
		{EndArray, nil, ""},
	},
	wantOut: "[\n    \t[\n    \t\t{\n    \t\t\t\"alpha\":\"bravo\", \n    \t\t\t\"fizz\":\"buzz\", \n    \t\t\t\"zap\":{\n    \t\t\t\t\"xxx\":1, \n    \t\t\t\t\"xxx\":22, \n    \t\t\t\t\"xxx\":333\n    \t\t\t}, \n    \t\t\t\"zip\":{\n    \t\t\t\t\"x\xfdx\":123, \n    \t\t\t\t\"x\xfex\":123, \n    \t\t\t\t\"x\xffx\":123\n    \t\t\t}\n    \t\t}\n    \t]\n    ]\n",
}, {
	name: jsontest.Name("Format/CanonicalizeRawInts"),
	opts: []Options{CanonicalizeRawInts(true), SpaceAfterComma(true)},
	calls: []encoderMethodCall{
		{Value(`[0.100,5.0,1E6,-9223372036854775808,-10,-1,-0,0,1,10,9223372036854775807]`), nil, ""},
	},
	wantOut: "[0.100, 5.0, 1E6, -9223372036854776000, -10, -1, 0, 0, 1, 10, 9223372036854776000]\n",
}, {
	name: jsontest.Name("Format/CanonicalizeRawFloats"),
	opts: []Options{CanonicalizeRawFloats(true), SpaceAfterComma(true)},
	calls: []encoderMethodCall{
		{Value(`[0.100,5.0,1E6,-9223372036854775808,-10,-1,-0,0,1,10,9223372036854775807]`), nil, ""},
	},
	wantOut: "[0.1, 5, 1000000, -9223372036854775808, -10, -1, 0, 0, 1, 10, 9223372036854775807]\n",
}, {
	name: jsontest.Name("ErrorPosition"),
	calls: []encoderMethodCall{
		{Value(` "a` + "\xff" + `0" `), E(jsonwire.ErrInvalidUTF8).withPos(` "a`, ""), ""},
		{String(`a` + "\xff" + `0`), E(jsonwire.ErrInvalidUTF8).withPos(``, ""), ""},
	},
}, {
	name: jsontest.Name("ErrorPosition/0"),
	calls: []encoderMethodCall{
		{Value(` [ "a` + "\xff" + `1" ] `), E(jsonwire.ErrInvalidUTF8).withPos(` [ "a`, "/0"), ""},
		{BeginArray, nil, ""},
		{Value(` "a` + "\xff" + `1" `), E(jsonwire.ErrInvalidUTF8).withPos(`[ "a`, "/0"), ""},
		{String(`a` + "\xff" + `1`), E(jsonwire.ErrInvalidUTF8).withPos(`[`, "/0"), ""},
	},
	wantOut: `[`,
}, {
	name: jsontest.Name("ErrorPosition/1"),
	calls: []encoderMethodCall{
		{Value(` [ "a1" , "b` + "\xff" + `1" ] `), E(jsonwire.ErrInvalidUTF8).withPos(` [ "a1" , "b`, "/1"), ""},
		{BeginArray, nil, ""},
		{String("a1"), nil, ""},
		{Value(` "b` + "\xff" + `1" `), E(jsonwire.ErrInvalidUTF8).withPos(`["a1", "b`, "/1"), ""},
		{String(`b` + "\xff" + `1`), E(jsonwire.ErrInvalidUTF8).withPos(`["a1",`, "/1"), ""},
	},
	wantOut: `["a1"`,
}, {
	name: jsontest.Name("ErrorPosition/0/0"),
	calls: []encoderMethodCall{
		{Value(` [ [ "a` + "\xff" + `2" ] ] `), E(jsonwire.ErrInvalidUTF8).withPos(` [ [ "a`, "/0/0"), ""},
		{BeginArray, nil, ""},
		{Value(` [ "a` + "\xff" + `2" ] `), E(jsonwire.ErrInvalidUTF8).withPos(`[ [ "a`, "/0/0"), ""},
		{BeginArray, nil, "/0"},
		{Value(` "a` + "\xff" + `2" `), E(jsonwire.ErrInvalidUTF8).withPos(`[[ "a`, "/0/0"), "/0"},
		{String(`a` + "\xff" + `2`), E(jsonwire.ErrInvalidUTF8).withPos(`[[`, "/0/0"), "/0"},
	},
	wantOut: `[[`,
}, {
	name: jsontest.Name("ErrorPosition/1/0"),
	calls: []encoderMethodCall{
		{Value(` [ "a1" , [ "a` + "\xff" + `2" ] ] `), E(jsonwire.ErrInvalidUTF8).withPos(` [ "a1" , [ "a`, "/1/0"), ""},
		{BeginArray, nil, ""},
		{String("a1"), nil, "/0"},
		{Value(` [ "a` + "\xff" + `2" ] `), E(jsonwire.ErrInvalidUTF8).withPos(`["a1", [ "a`, "/1/0"), ""},
		{BeginArray, nil, "/1"},
		{Value(` "a` + "\xff" + `2" `), E(jsonwire.ErrInvalidUTF8).withPos(`["a1",[ "a`, "/1/0"), "/1"},
		{String(`a` + "\xff" + `2`), E(jsonwire.ErrInvalidUTF8).withPos(`["a1",[`, "/1/0"), "/1"},
	},
	wantOut: `["a1",[`,
}, {
	name: jsontest.Name("ErrorPosition/0/1"),
	calls: []encoderMethodCall{
		{Value(` [ [ "a2" , "b` + "\xff" + `2" ] ] `), E(jsonwire.ErrInvalidUTF8).withPos(` [ [ "a2" , "b`, "/0/1"), ""},
		{BeginArray, nil, ""},
		{Value(` [ "a2" , "b` + "\xff" + `2" ] `), E(jsonwire.ErrInvalidUTF8).withPos(`[ [ "a2" , "b`, "/0/1"), ""},
		{BeginArray, nil, "/0"},
		{String("a2"), nil, "/0/0"},
		{Value(` "b` + "\xff" + `2" `), E(jsonwire.ErrInvalidUTF8).withPos(`[["a2", "b`, "/0/1"), "/0/0"},
		{String(`b` + "\xff" + `2`), E(jsonwire.ErrInvalidUTF8).withPos(`[["a2",`, "/0/1"), "/0/0"},
	},
	wantOut: `[["a2"`,
}, {
	name: jsontest.Name("ErrorPosition/1/1"),
	calls: []encoderMethodCall{
		{Value(` [ "a1" , [ "a2" , "b` + "\xff" + `2" ] ] `), E(jsonwire.ErrInvalidUTF8).withPos(` [ "a1" , [ "a2" , "b`, "/1/1"), ""},
		{BeginArray, nil, ""},
		{String("a1"), nil, "/0"},
		{Value(` [ "a2" , "b` + "\xff" + `2" ] `), E(jsonwire.ErrInvalidUTF8).withPos(`["a1", [ "a2" , "b`, "/1/1"), ""},
		{BeginArray, nil, "/1"},
		{String("a2"), nil, "/1/0"},
		{Value(` "b` + "\xff" + `2" `), E(jsonwire.ErrInvalidUTF8).withPos(`["a1",["a2", "b`, "/1/1"), "/1/0"},
		{String(`b` + "\xff" + `2`), E(jsonwire.ErrInvalidUTF8).withPos(`["a1",["a2",`, "/1/1"), "/1/0"},
	},
	wantOut: `["a1",["a2"`,
}, {
	name: jsontest.Name("ErrorPosition/a1-"),
	calls: []encoderMethodCall{
		{Value(` { "a` + "\xff" + `1" : "b1" } `), E(jsonwire.ErrInvalidUTF8).withPos(` { "a`, ""), ""},
		{BeginObject, nil, ""},
		{Value(` "a` + "\xff" + `1" `), E(jsonwire.ErrInvalidUTF8).withPos(`{ "a`, ""), ""},
		{String(`a` + "\xff" + `1`), E(jsonwire.ErrInvalidUTF8).withPos(`{`, ""), ""},
	},
	wantOut: `{`,
}, {
	name: jsontest.Name("ErrorPosition/a1"),
	calls: []encoderMethodCall{
		{Value(` { "a1" : "b` + "\xff" + `1" } `), E(jsonwire.ErrInvalidUTF8).withPos(` { "a1" : "b`, "/a1"), ""},
		{BeginObject, nil, ""},
		{String("a1"), nil, "/a1"},
		{Value(` "b` + "\xff" + `1" `), E(jsonwire.ErrInvalidUTF8).withPos(`{"a1": "b`, "/a1"), ""},
		{String(`b` + "\xff" + `1`), E(jsonwire.ErrInvalidUTF8).withPos(`{"a1":`, "/a1"), ""},
	},
	wantOut: `{"a1"`,
}, {
	name: jsontest.Name("ErrorPosition/c1-"),
	calls: []encoderMethodCall{
		{Value(` { "a1" : "b1" , "c` + "\xff" + `1" : "d1" } `), E(jsonwire.ErrInvalidUTF8).withPos(` { "a1" : "b1" , "c`, ""), ""},
		{BeginObject, nil, ""},
		{String("a1"), nil, "/a1"},
		{String("b1"), nil, "/a1"},
		{Value(` "c` + "\xff" + `1" `), E(jsonwire.ErrInvalidUTF8).withPos(`{"a1":"b1": "c`, ""), "/a1"},
		{String(`c` + "\xff" + `1`), E(jsonwire.ErrInvalidUTF8).withPos(`{"a1":"b1":`, ""), "/a1"},
	},
	wantOut: `{"a1":"b1"`,
}, {
	name: jsontest.Name("ErrorPosition/c1"),
	calls: []encoderMethodCall{
		{Value(` { "a1" : "b1" , "c1" : "d` + "\xff" + `1" } `), E(jsonwire.ErrInvalidUTF8).withPos(` { "a1" : "b1" , "c1" : "d`, "/c1"), ""},
		{BeginObject, nil, ""},
		{String("a1"), nil, "/a1"},
		{String("b1"), nil, "/a1"},
		{String("c1"), nil, "/c1"},
		{Value(` "d` + "\xff" + `1" `), E(jsonwire.ErrInvalidUTF8).withPos(`{"a1":"b1":"c1": "d`, "/c1"), "/c1"},
		{String(`d` + "\xff" + `1`), E(jsonwire.ErrInvalidUTF8).withPos(`{"a1":"b1":"c1":`, "/c1"), "/c1"},
	},
	wantOut: `{"a1":"b1","c1"`,
}, {
	name: jsontest.Name("ErrorPosition/a1/a2-"),
	calls: []encoderMethodCall{
		{Value(` { "a1" : { "a` + "\xff" + `2" : "b2" } } `), E(jsonwire.ErrInvalidUTF8).withPos(` { "a1" : { "a`, "/a1"), ""},
		{BeginObject, nil, ""},
		{String("a1"), nil, "/a1"},
		{Value(` { "a` + "\xff" + `2" : "b2" } `), E(jsonwire.ErrInvalidUTF8).withPos(`{"a1": { "a`, "/a1"), ""},
		{BeginObject, nil, "/a1"},
		{Value(` "a` + "\xff" + `2" `), E(jsonwire.ErrInvalidUTF8).withPos(`{"a1":{ "a`, "/a1"), "/a1"},
		{String(`a` + "\xff" + `2`), E(jsonwire.ErrInvalidUTF8).withPos(`{"a1":{`, "/a1"), "/a1"},
	},
	wantOut: `{"a1":{`,
}, {
	name: jsontest.Name("ErrorPosition/a1/a2"),
	calls: []encoderMethodCall{
		{Value(` { "a1" : { "a2" : "b` + "\xff" + `2" } } `), E(jsonwire.ErrInvalidUTF8).withPos(` { "a1" : { "a2" : "b`, "/a1/a2"), ""},
		{BeginObject, nil, ""},
		{String("a1"), nil, "/a1"},
		{Value(` { "a2" : "b` + "\xff" + `2" } `), E(jsonwire.ErrInvalidUTF8).withPos(`{"a1": { "a2" : "b`, "/a1/a2"), ""},
		{BeginObject, nil, "/a1"},
		{String("a2"), nil, "/a1/a2"},
		{Value(` "b` + "\xff" + `2" `), E(jsonwire.ErrInvalidUTF8).withPos(`{"a1":{"a2": "b`, "/a1/a2"), "/a1/a2"},
		{String(`b` + "\xff" + `2`), E(jsonwire.ErrInvalidUTF8).withPos(`{"a1":{"a2":`, "/a1/a2"), "/a1/a2"},
	},
	wantOut: `{"a1":{"a2"`,
}, {
	name: jsontest.Name("ErrorPosition/a1/c2-"),
	calls: []encoderMethodCall{
		{Value(` { "a1" : { "a2" : "b2" , "c` + "\xff" + `2" : "d2" } } `), E(jsonwire.ErrInvalidUTF8).withPos(` { "a1" : { "a2" : "b2" , "c`, "/a1"), ""},
		{BeginObject, nil, ""},
		{String("a1"), nil, "/a1"},
		{BeginObject, nil, "/a1"},
		{String("a2"), nil, "/a1/a2"},
		{String("b2"), nil, "/a1/a2"},
		{Value(` "c` + "\xff" + `2" `), E(jsonwire.ErrInvalidUTF8).withPos(`{"a1":{"a2":"b2", "c`, "/a1"), "/a1/a2"},
		{String(`c` + "\xff" + `2`), E(jsonwire.ErrInvalidUTF8).withPos(`{"a1":{"a2":"b2",`, "/a1"), "/a1/a2"},
	},
	wantOut: `{"a1":{"a2":"b2"`,
}, {
	name: jsontest.Name("ErrorPosition/a1/c2"),
	calls: []encoderMethodCall{
		{Value(` { "a1" : { "a2" : "b2" , "c2" : "d` + "\xff" + `2" } } `), E(jsonwire.ErrInvalidUTF8).withPos(` { "a1" : { "a2" : "b2" , "c2" : "d`, "/a1/c2"), ""},
		{BeginObject, nil, ""},
		{String("a1"), nil, "/a1"},
		{Value(` { "a2" : "b2" , "c2" : "d` + "\xff" + `2" } `), E(jsonwire.ErrInvalidUTF8).withPos(`{"a1": { "a2" : "b2" , "c2" : "d`, "/a1/c2"), ""},
		{BeginObject, nil, ""},
		{String("a2"), nil, "/a1/a2"},
		{String("b2"), nil, "/a1/a2"},
		{String("c2"), nil, "/a1/c2"},
		{Value(` "d` + "\xff" + `2" `), E(jsonwire.ErrInvalidUTF8).withPos(`{"a1":{"a2":"b2","c2": "d`, "/a1/c2"), "/a1/c2"},
		{String(`d` + "\xff" + `2`), E(jsonwire.ErrInvalidUTF8).withPos(`{"a1":{"a2":"b2","c2":`, "/a1/c2"), "/a1/c2"},
	},
	wantOut: `{"a1":{"a2":"b2","c2"`,
}, {
	name: jsontest.Name("ErrorPosition/1/a2"),
	calls: []encoderMethodCall{
		{Value(` [ "a1" , { "a2" : "b` + "\xff" + `2" } ] `), E(jsonwire.ErrInvalidUTF8).withPos(` [ "a1" , { "a2" : "b`, "/1/a2"), ""},
		{BeginArray, nil, ""},
		{String("a1"), nil, "/0"},
		{Value(` { "a2" : "b` + "\xff" + `2" } `), E(jsonwire.ErrInvalidUTF8).withPos(`["a1", { "a2" : "b`, "/1/a2"), ""},
		{BeginObject, nil, "/1"},
		{String("a2"), nil, "/1/a2"},
		{Value(` "b` + "\xff" + `2" `), E(jsonwire.ErrInvalidUTF8).withPos(`["a1",{"a2": "b`, "/1/a2"), "/1/a2"},
		{String(`b` + "\xff" + `2`), E(jsonwire.ErrInvalidUTF8).withPos(`["a1",{"a2":`, "/1/a2"), "/1/a2"},
	},
	wantOut: `["a1",{"a2"`,
}, {
	name: jsontest.Name("ErrorPosition/c1/1"),
	calls: []encoderMethodCall{
		{Value(` { "a1" : "b1" , "c1" : [ "a2" , "b` + "\xff" + `2" ] } `), E(jsonwire.ErrInvalidUTF8).withPos(` { "a1" : "b1" , "c1" : [ "a2" , "b`, "/c1/1"), ""},
		{BeginObject, nil, ""},
		{String("a1"), nil, "/a1"},
		{String("b1"), nil, "/a1"},
		{String("c1"), nil, "/c1"},
		{Value(` [ "a2" , "b` + "\xff" + `2" ] `), E(jsonwire.ErrInvalidUTF8).withPos(`{"a1":"b1","c1": [ "a2" , "b`, "/c1/1"), ""},
		{BeginArray, nil, "/c1"},
		{String("a2"), nil, "/c1/0"},
		{Value(` "b` + "\xff" + `2" `), E(jsonwire.ErrInvalidUTF8).withPos(`{"a1":"b1","c1":["a2", "b`, "/c1/1"), "/c1/0"},
		{String(`b` + "\xff" + `2`), E(jsonwire.ErrInvalidUTF8).withPos(`{"a1":"b1","c1":["a2",`, "/c1/1"), "/c1/0"},
	},
	wantOut: `{"a1":"b1","c1":["a2"`,
}, {
	name: jsontest.Name("ErrorPosition/0/a1/1/c3/1"),
	calls: []encoderMethodCall{
		{Value(` [ { "a1" : [ "a2" , { "a3" : "b3" , "c3" : [ "a4" , "b` + "\xff" + `4" ] } ] } ] `), E(jsonwire.ErrInvalidUTF8).withPos(` [ { "a1" : [ "a2" , { "a3" : "b3" , "c3" : [ "a4" , "b`, "/0/a1/1/c3/1"), ""},
		{BeginArray, nil, ""},
		{Value(` { "a1" : [ "a2" , { "a3" : "b3" , "c3" : [ "a4" , "b` + "\xff" + `4" ] } ] } `), E(jsonwire.ErrInvalidUTF8).withPos(`[ { "a1" : [ "a2" , { "a3" : "b3" , "c3" : [ "a4" , "b`, "/0/a1/1/c3/1"), ""},
		{BeginObject, nil, "/0"},
		{String("a1"), nil, "/0/a1"},
		{Value(` [ "a2" , { "a3" : "b3" , "c3" : [ "a4" , "b` + "\xff" + `4" ] } ] `), E(jsonwire.ErrInvalidUTF8).withPos(`[{"a1": [ "a2" , { "a3" : "b3" , "c3" : [ "a4" , "b`, "/0/a1/1/c3/1"), ""},
		{BeginArray, nil, ""},
		{String("a2"), nil, "/0/a1/0"},
		{Value(` { "a3" : "b3" , "c3" : [ "a4" , "b` + "\xff" + `4" ] } `), E(jsonwire.ErrInvalidUTF8).withPos(`[{"a1":["a2", { "a3" : "b3" , "c3" : [ "a4" , "b`, "/0/a1/1/c3/1"), ""},
		{BeginObject, nil, "/0/a1/1"},
		{String("a3"), nil, "/0/a1/1/a3"},
		{String("b3"), nil, "/0/a1/1/a3"},
		{String("c3"), nil, "/0/a1/1/c3"},
		{Value(` [ "a4" , "b` + "\xff" + `4" ] `), E(jsonwire.ErrInvalidUTF8).withPos(`[{"a1":["a2",{"a3":"b3","c3": [ "a4" , "b`, "/0/a1/1/c3/1"), ""},
		{BeginArray, nil, "/0/a1/1/c3"},
		{String("a4"), nil, "/0/a1/1/c3/0"},
		{Value(` "b` + "\xff" + `4" `), E(jsonwire.ErrInvalidUTF8).withPos(`[{"a1":["a2",{"a3":"b3","c3":["a4", "b`, "/0/a1/1/c3/1"), "/0/a1/1/c3/0"},
		{String(`b` + "\xff" + `4`), E(jsonwire.ErrInvalidUTF8).withPos(`[{"a1":["a2",{"a3":"b3","c3":["a4",`, "/0/a1/1/c3/1"), "/0/a1/1/c3/0"},
	},
	wantOut: `[{"a1":["a2",{"a3":"b3","c3":["a4"`,
}}

// TestEncoderErrors test that Encoder errors occur when we expect and
// leaves the Encoder in a consistent state.
func TestEncoderErrors(t *testing.T) {
	for _, td := range encoderErrorTestdata {
		t.Run(path.Join(td.name.Name), func(t *testing.T) {
			testEncoderErrors(t, td.name.Where, td.opts, td.calls, td.wantOut)
		})
	}
}
func testEncoderErrors(t *testing.T, where jsontest.CasePos, opts []Options, calls []encoderMethodCall, wantOut string) {
	dst := new(bytes.Buffer)
	enc := NewEncoder(dst, opts...)
	for i, call := range calls {
		var gotErr error
		switch tokVal := call.in.(type) {
		case Token:
			gotErr = enc.WriteToken(tokVal)
		case Value:
			gotErr = enc.WriteValue(tokVal)
		}
		if !equalError(gotErr, call.wantErr) {
			t.Fatalf("%s: %d: error mismatch:\ngot  %v\nwant %v", where, i, gotErr, call.wantErr)
		}
		if call.wantPointer != "" {
			gotPointer := enc.StackPointer()
			if gotPointer != call.wantPointer {
				t.Fatalf("%s: %d: Encoder.StackPointer = %s, want %s", where, i, gotPointer, call.wantPointer)
			}
		}
	}
	gotOut := dst.String() + string(enc.s.unflushedBuffer())
	if gotOut != wantOut {
		t.Fatalf("%s: output mismatch:\ngot  %q\nwant %q", where, gotOut, wantOut)
	}
	gotOffset := int(enc.OutputOffset())
	wantOffset := len(wantOut)
	if gotOffset != wantOffset {
		t.Fatalf("%s: Encoder.OutputOffset = %v, want %v", where, gotOffset, wantOffset)
	}
}

// TestEncoderReset tests that the encoder preserves its internal
// buffer between Reset calls to avoid frequent allocations when reusing the encoder.
// It ensures that the buffer capacity is maintained while avoiding aliasing
// issues with [bytes.Buffer].
func TestEncoderReset(t *testing.T) {
	// Create an encoder with a reasonably large JSON input to ensure buffer growth.
	largeJSON := `{"key1":"value1","key2":"value2","key3":"value3","key4":"value4","key5":"value5"}` + "\n"
	bb := new(bytes.Buffer)
	enc := NewEncoder(struct{ io.Writer }{bb}) // mask out underlying [bytes.Buffer]

	t.Run("Test capacity preservation", func(t *testing.T) {
		// Write the first JSON value to grow the internal buffer.
		err := enc.WriteValue(append(enc.AvailableBuffer(), largeJSON...))
		if err != nil {
			t.Fatalf("first WriteValue failed: %v", err)
		}
		if bb.String() != largeJSON {
			t.Fatalf("first WriteValue = %q, want %q", bb.String(), largeJSON)
		}

		// Get the buffer capacity after first use.
		initialCapacity := cap(enc.s.Buf)
		initialCacheCapacity := cap(enc.s.availBuffer)
		if initialCapacity == 0 {
			t.Fatalf("expected non-zero buffer capacity after first use")
		}
		if initialCacheCapacity == 0 {
			t.Fatalf("expected non-zero cache capacity after first use")
		}

		// Reset with a new writer - this should preserve the buffer capacity.
		bb.Reset()
		enc.Reset(struct{ io.Writer }{bb})

		// Verify the buffer capacity is preserved (or at least not smaller).
		preservedCapacity := cap(enc.s.Buf)
		if preservedCapacity < initialCapacity {
			t.Fatalf("buffer capacity reduced after Reset: got %d, want at least %d", preservedCapacity, initialCapacity)
		}
		preservedCacheCapacity := cap(enc.s.availBuffer)
		if preservedCacheCapacity < initialCacheCapacity {
			t.Fatalf("cache capacity reduced after Reset: got %d, want at least %d", preservedCapacity, initialCapacity)
		}

		// Write the second JSON value to ensure the encoder still works correctly.
		err = enc.WriteValue(append(enc.AvailableBuffer(), largeJSON...))
		if err != nil {
			t.Fatalf("second WriteValue failed: %v", err)
		}
		if bb.String() != largeJSON {
			t.Fatalf("second WriteValue = %q, want %q", bb.String(), largeJSON)
		}
	})

	t.Run("Test aliasing with bytes.Buffer", func(t *testing.T) {
		// Test with bytes.Buffer to verify proper aliasing behavior.
		bb.Reset()
		enc.Reset(bb)

		// Write the third JSON value to ensure functionality with bytes.Buffer.
		err := enc.WriteValue([]byte(largeJSON))
		if err != nil {
			t.Fatalf("fourth WriteValue failed: %v", err)
		}
		if bb.String() != largeJSON {
			t.Fatalf("fourth WriteValue = %q, want %q", bb.String(), largeJSON)
		}
		// The encoder buffer should alias bytes.Buffer's internal buffer.
		if cap(enc.s.Buf) == 0 || cap(bb.AvailableBuffer()) == 0 || &enc.s.Buf[:1][0] != &bb.AvailableBuffer()[:1][0] {
			t.Fatalf("encoder buffer does not alias bytes.Buffer")
		}
	})

	t.Run("Test aliasing removed after Reset", func(t *testing.T) {
		// Reset with a new reader and verify the buffer is not aliased.
		bb.Reset()
		enc.Reset(struct{ io.Writer }{bb})
		err := enc.WriteValue([]byte(largeJSON))
		if err != nil {
			t.Fatalf("fifth WriteValue failed: %v", err)
		}
		if bb.String() != largeJSON {
			t.Fatalf("fourth WriteValue = %q, want %q", bb.String(), largeJSON)
		}

		// The encoder buffer should not alias the bytes.Buffer's internal buffer.
		if cap(enc.s.Buf) == 0 || cap(bb.AvailableBuffer()) == 0 || &enc.s.Buf[:1][0] == &bb.AvailableBuffer()[:1][0] {
			t.Fatalf("encoder buffer aliases bytes.Buffer")
		}
	})
}
