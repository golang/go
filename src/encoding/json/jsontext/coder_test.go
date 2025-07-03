// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jsonv2

package jsontext

import (
	"bytes"
	"errors"
	"io"
	"math"
	"math/rand"
	"path"
	"reflect"
	"strings"
	"testing"

	"encoding/json/internal/jsontest"
	"encoding/json/internal/jsonwire"
)

func E(err error) *SyntacticError {
	return &SyntacticError{Err: err}
}

func newInvalidCharacterError(prefix, where string) *SyntacticError {
	return E(jsonwire.NewInvalidCharacterError(prefix, where))
}

func newInvalidEscapeSequenceError(what string) *SyntacticError {
	return E(jsonwire.NewInvalidEscapeSequenceError(what))
}

func (e *SyntacticError) withPos(prefix string, pointer Pointer) *SyntacticError {
	e.ByteOffset = int64(len(prefix))
	e.JSONPointer = pointer
	return e
}

func equalError(x, y error) bool {
	return reflect.DeepEqual(x, y)
}

var (
	zeroToken Token
	zeroValue Value
)

// tokOrVal is either a Token or a Value.
type tokOrVal interface{ Kind() Kind }

type coderTestdataEntry struct {
	name             jsontest.CaseName
	in               string
	outCompacted     string
	outEscaped       string // outCompacted if empty; escapes all runes in a string
	outIndented      string // outCompacted if empty; uses "  " for indent prefix and "\t" for indent
	outCanonicalized string // outCompacted if empty
	tokens           []Token
	pointers         []Pointer
}

var coderTestdata = []coderTestdataEntry{{
	name:         jsontest.Name("Null"),
	in:           ` null `,
	outCompacted: `null`,
	tokens:       []Token{Null},
	pointers:     []Pointer{""},
}, {
	name:         jsontest.Name("False"),
	in:           ` false `,
	outCompacted: `false`,
	tokens:       []Token{False},
}, {
	name:         jsontest.Name("True"),
	in:           ` true `,
	outCompacted: `true`,
	tokens:       []Token{True},
}, {
	name:         jsontest.Name("EmptyString"),
	in:           ` "" `,
	outCompacted: `""`,
	tokens:       []Token{String("")},
}, {
	name:         jsontest.Name("SimpleString"),
	in:           ` "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" `,
	outCompacted: `"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"`,
	outEscaped:   `"\u0061\u0062\u0063\u0064\u0065\u0066\u0067\u0068\u0069\u006a\u006b\u006c\u006d\u006e\u006f\u0070\u0071\u0072\u0073\u0074\u0075\u0076\u0077\u0078\u0079\u007a\u0041\u0042\u0043\u0044\u0045\u0046\u0047\u0048\u0049\u004a\u004b\u004c\u004d\u004e\u004f\u0050\u0051\u0052\u0053\u0054\u0055\u0056\u0057\u0058\u0059\u005a"`,
	tokens:       []Token{String("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")},
}, {
	name:             jsontest.Name("ComplicatedString"),
	in:               " \"Hello, 世界 🌟★☆✩🌠 " + "\u0080\u00f6\u20ac\ud799\ue000\ufb33\ufffd\U0001f602" + ` \ud800\udead \"\\\/\b\f\n\r\t \u0022\u005c\u002f\u0008\u000c\u000a\u000d\u0009" `,
	outCompacted:     "\"Hello, 世界 🌟★☆✩🌠 " + "\u0080\u00f6\u20ac\ud799\ue000\ufb33\ufffd\U0001f602" + " 𐊭 \\\"\\\\/\\b\\f\\n\\r\\t \\\"\\\\/\\b\\f\\n\\r\\t\"",
	outEscaped:       `"\u0048\u0065\u006c\u006c\u006f\u002c\u0020\u4e16\u754c\u0020\ud83c\udf1f\u2605\u2606\u2729\ud83c\udf20\u0020\u0080\u00f6\u20ac\ud799\ue000\ufb33\ufffd\ud83d\ude02\u0020\ud800\udead\u0020\u0022\u005c\u002f\u0008\u000c\u000a\u000d\u0009\u0020\u0022\u005c\u002f\u0008\u000c\u000a\u000d\u0009"`,
	outCanonicalized: `"Hello, 世界 🌟★☆✩🌠 ö€힙דּ�😂 𐊭 \"\\/\b\f\n\r\t \"\\/\b\f\n\r\t"`,
	tokens:           []Token{rawToken("\"Hello, 世界 🌟★☆✩🌠 " + "\u0080\u00f6\u20ac\ud799\ue000\ufb33\ufffd\U0001f602" + " 𐊭 \\\"\\\\/\\b\\f\\n\\r\\t \\\"\\\\/\\b\\f\\n\\r\\t\"")},
}, {
	name:         jsontest.Name("ZeroNumber"),
	in:           ` 0 `,
	outCompacted: `0`,
	tokens:       []Token{Uint(0)},
}, {
	name:         jsontest.Name("SimpleNumber"),
	in:           ` 123456789 `,
	outCompacted: `123456789`,
	tokens:       []Token{Uint(123456789)},
}, {
	name:         jsontest.Name("NegativeNumber"),
	in:           ` -123456789 `,
	outCompacted: `-123456789`,
	tokens:       []Token{Int(-123456789)},
}, {
	name:         jsontest.Name("FractionalNumber"),
	in:           " 0.123456789 ",
	outCompacted: `0.123456789`,
	tokens:       []Token{Float(0.123456789)},
}, {
	name:             jsontest.Name("ExponentNumber"),
	in:               " 0e12456789 ",
	outCompacted:     `0e12456789`,
	outCanonicalized: `0`,
	tokens:           []Token{rawToken(`0e12456789`)},
}, {
	name:             jsontest.Name("ExponentNumberP"),
	in:               " 0e+12456789 ",
	outCompacted:     `0e+12456789`,
	outCanonicalized: `0`,
	tokens:           []Token{rawToken(`0e+12456789`)},
}, {
	name:             jsontest.Name("ExponentNumberN"),
	in:               " 0e-12456789 ",
	outCompacted:     `0e-12456789`,
	outCanonicalized: `0`,
	tokens:           []Token{rawToken(`0e-12456789`)},
}, {
	name:             jsontest.Name("ComplicatedNumber"),
	in:               ` -123456789.987654321E+0123456789 `,
	outCompacted:     `-123456789.987654321E+0123456789`,
	outCanonicalized: `-1.7976931348623157e+308`,
	tokens:           []Token{rawToken(`-123456789.987654321E+0123456789`)},
}, {
	name: jsontest.Name("Numbers"),
	in: ` [
		0, -0, 0.0, -0.0, 1.00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001, 1e1000,
		-5e-324, 1e+100, 1.7976931348623157e+308,
		9007199254740990, 9007199254740991, 9007199254740992, 9007199254740993, 9007199254740994,
		-9223372036854775808, 9223372036854775807, 0, 18446744073709551615
	] `,
	outCompacted: "[0,-0,0.0,-0.0,1.00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001,1e1000,-5e-324,1e+100,1.7976931348623157e+308,9007199254740990,9007199254740991,9007199254740992,9007199254740993,9007199254740994,-9223372036854775808,9223372036854775807,0,18446744073709551615]",
	outIndented: `[
	    0,
	    -0,
	    0.0,
	    -0.0,
	    1.00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001,
	    1e1000,
	    -5e-324,
	    1e+100,
	    1.7976931348623157e+308,
	    9007199254740990,
	    9007199254740991,
	    9007199254740992,
	    9007199254740993,
	    9007199254740994,
	    -9223372036854775808,
	    9223372036854775807,
	    0,
	    18446744073709551615
	]`,
	outCanonicalized: `[0,0,0,0,1,1.7976931348623157e+308,-5e-324,1e+100,1.7976931348623157e+308,9007199254740990,9007199254740991,9007199254740992,9007199254740992,9007199254740994,-9223372036854776000,9223372036854776000,0,18446744073709552000]`,
	tokens: []Token{
		BeginArray,
		Float(0), Float(math.Copysign(0, -1)), rawToken(`0.0`), rawToken(`-0.0`), rawToken(`1.00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001`), rawToken(`1e1000`),
		Float(-5e-324), Float(1e100), Float(1.7976931348623157e+308),
		Float(9007199254740990), Float(9007199254740991), Float(9007199254740992), rawToken(`9007199254740993`), rawToken(`9007199254740994`),
		Int(minInt64), Int(maxInt64), Uint(minUint64), Uint(maxUint64),
		EndArray,
	},
	pointers: []Pointer{
		"", "/0", "/1", "/2", "/3", "/4", "/5", "/6", "/7", "/8", "/9", "/10", "/11", "/12", "/13", "/14", "/15", "/16", "/17", "",
	},
}, {
	name:         jsontest.Name("ObjectN0"),
	in:           ` { } `,
	outCompacted: `{}`,
	tokens:       []Token{BeginObject, EndObject},
	pointers:     []Pointer{"", ""},
}, {
	name:         jsontest.Name("ObjectN1"),
	in:           ` { "0" : 0 } `,
	outCompacted: `{"0":0}`,
	outEscaped:   `{"\u0030":0}`,
	outIndented: `{
	    "0": 0
	}`,
	tokens:   []Token{BeginObject, String("0"), Uint(0), EndObject},
	pointers: []Pointer{"", "/0", "/0", ""},
}, {
	name:         jsontest.Name("ObjectN2"),
	in:           ` { "0" : 0 , "1" : 1 } `,
	outCompacted: `{"0":0,"1":1}`,
	outEscaped:   `{"\u0030":0,"\u0031":1}`,
	outIndented: `{
	    "0": 0,
	    "1": 1
	}`,
	tokens:   []Token{BeginObject, String("0"), Uint(0), String("1"), Uint(1), EndObject},
	pointers: []Pointer{"", "/0", "/0", "/1", "/1", ""},
}, {
	name:         jsontest.Name("ObjectNested"),
	in:           ` { "0" : { "1" : { "2" : { "3" : { "4" : {  } } } } } } `,
	outCompacted: `{"0":{"1":{"2":{"3":{"4":{}}}}}}`,
	outEscaped:   `{"\u0030":{"\u0031":{"\u0032":{"\u0033":{"\u0034":{}}}}}}`,
	outIndented: `{
	    "0": {
	        "1": {
	            "2": {
	                "3": {
	                    "4": {}
	                }
	            }
	        }
	    }
	}`,
	tokens: []Token{BeginObject, String("0"), BeginObject, String("1"), BeginObject, String("2"), BeginObject, String("3"), BeginObject, String("4"), BeginObject, EndObject, EndObject, EndObject, EndObject, EndObject, EndObject},
	pointers: []Pointer{
		"",
		"/0", "/0",
		"/0/1", "/0/1",
		"/0/1/2", "/0/1/2",
		"/0/1/2/3", "/0/1/2/3",
		"/0/1/2/3/4", "/0/1/2/3/4",
		"/0/1/2/3/4",
		"/0/1/2/3",
		"/0/1/2",
		"/0/1",
		"/0",
		"",
	},
}, {
	name: jsontest.Name("ObjectSuperNested"),
	in: `{"": {
		"44444": {
			"6666666":  "ccccccc",
			"77777777": "bb",
			"555555":   "aaaa"
		},
		"0": {
			"3333": "bbb",
			"11":   "",
			"222":  "aaaaa"
		}
	}}`,
	outCompacted: `{"":{"44444":{"6666666":"ccccccc","77777777":"bb","555555":"aaaa"},"0":{"3333":"bbb","11":"","222":"aaaaa"}}}`,
	outEscaped:   `{"":{"\u0034\u0034\u0034\u0034\u0034":{"\u0036\u0036\u0036\u0036\u0036\u0036\u0036":"\u0063\u0063\u0063\u0063\u0063\u0063\u0063","\u0037\u0037\u0037\u0037\u0037\u0037\u0037\u0037":"\u0062\u0062","\u0035\u0035\u0035\u0035\u0035\u0035":"\u0061\u0061\u0061\u0061"},"\u0030":{"\u0033\u0033\u0033\u0033":"\u0062\u0062\u0062","\u0031\u0031":"","\u0032\u0032\u0032":"\u0061\u0061\u0061\u0061\u0061"}}}`,
	outIndented: `{
	    "": {
	        "44444": {
	            "6666666": "ccccccc",
	            "77777777": "bb",
	            "555555": "aaaa"
	        },
	        "0": {
	            "3333": "bbb",
	            "11": "",
	            "222": "aaaaa"
	        }
	    }
	}`,
	outCanonicalized: `{"":{"0":{"11":"","222":"aaaaa","3333":"bbb"},"44444":{"555555":"aaaa","6666666":"ccccccc","77777777":"bb"}}}`,
	tokens: []Token{
		BeginObject,
		String(""),
		BeginObject,
		String("44444"),
		BeginObject,
		String("6666666"), String("ccccccc"),
		String("77777777"), String("bb"),
		String("555555"), String("aaaa"),
		EndObject,
		String("0"),
		BeginObject,
		String("3333"), String("bbb"),
		String("11"), String(""),
		String("222"), String("aaaaa"),
		EndObject,
		EndObject,
		EndObject,
	},
	pointers: []Pointer{
		"",
		"/", "/",
		"//44444", "//44444",
		"//44444/6666666", "//44444/6666666",
		"//44444/77777777", "//44444/77777777",
		"//44444/555555", "//44444/555555",
		"//44444",
		"//0", "//0",
		"//0/3333", "//0/3333",
		"//0/11", "//0/11",
		"//0/222", "//0/222",
		"//0",
		"/",
		"",
	},
}, {
	name:         jsontest.Name("ArrayN0"),
	in:           ` [ ] `,
	outCompacted: `[]`,
	tokens:       []Token{BeginArray, EndArray},
	pointers:     []Pointer{"", ""},
}, {
	name:         jsontest.Name("ArrayN1"),
	in:           ` [ 0 ] `,
	outCompacted: `[0]`,
	outIndented: `[
	    0
	]`,
	tokens:   []Token{BeginArray, Uint(0), EndArray},
	pointers: []Pointer{"", "/0", ""},
}, {
	name:         jsontest.Name("ArrayN2"),
	in:           ` [ 0 , 1 ] `,
	outCompacted: `[0,1]`,
	outIndented: `[
	    0,
	    1
	]`,
	tokens: []Token{BeginArray, Uint(0), Uint(1), EndArray},
}, {
	name:         jsontest.Name("ArrayNested"),
	in:           ` [ [ [ [ [ ] ] ] ] ] `,
	outCompacted: `[[[[[]]]]]`,
	outIndented: `[
	    [
	        [
	            [
	                []
	            ]
	        ]
	    ]
	]`,
	tokens: []Token{BeginArray, BeginArray, BeginArray, BeginArray, BeginArray, EndArray, EndArray, EndArray, EndArray, EndArray},
	pointers: []Pointer{
		"",
		"/0",
		"/0/0",
		"/0/0/0",
		"/0/0/0/0",
		"/0/0/0/0",
		"/0/0/0",
		"/0/0",
		"/0",
		"",
	},
}, {
	name: jsontest.Name("Everything"),
	in: ` {
		"literals" : [ null , false , true ],
		"string" : "Hello, 世界" ,
		"number" : 3.14159 ,
		"arrayN0" : [ ] ,
		"arrayN1" : [ 0 ] ,
		"arrayN2" : [ 0 , 1 ] ,
		"objectN0" : { } ,
		"objectN1" : { "0" : 0 } ,
		"objectN2" : { "0" : 0 , "1" : 1 }
	} `,
	outCompacted: `{"literals":[null,false,true],"string":"Hello, 世界","number":3.14159,"arrayN0":[],"arrayN1":[0],"arrayN2":[0,1],"objectN0":{},"objectN1":{"0":0},"objectN2":{"0":0,"1":1}}`,
	outEscaped:   `{"\u006c\u0069\u0074\u0065\u0072\u0061\u006c\u0073":[null,false,true],"\u0073\u0074\u0072\u0069\u006e\u0067":"\u0048\u0065\u006c\u006c\u006f\u002c\u0020\u4e16\u754c","\u006e\u0075\u006d\u0062\u0065\u0072":3.14159,"\u0061\u0072\u0072\u0061\u0079\u004e\u0030":[],"\u0061\u0072\u0072\u0061\u0079\u004e\u0031":[0],"\u0061\u0072\u0072\u0061\u0079\u004e\u0032":[0,1],"\u006f\u0062\u006a\u0065\u0063\u0074\u004e\u0030":{},"\u006f\u0062\u006a\u0065\u0063\u0074\u004e\u0031":{"\u0030":0},"\u006f\u0062\u006a\u0065\u0063\u0074\u004e\u0032":{"\u0030":0,"\u0031":1}}`,
	outIndented: `{
	    "literals": [
	        null,
	        false,
	        true
	    ],
	    "string": "Hello, 世界",
	    "number": 3.14159,
	    "arrayN0": [],
	    "arrayN1": [
	        0
	    ],
	    "arrayN2": [
	        0,
	        1
	    ],
	    "objectN0": {},
	    "objectN1": {
	        "0": 0
	    },
	    "objectN2": {
	        "0": 0,
	        "1": 1
	    }
	}`,
	outCanonicalized: `{"arrayN0":[],"arrayN1":[0],"arrayN2":[0,1],"literals":[null,false,true],"number":3.14159,"objectN0":{},"objectN1":{"0":0},"objectN2":{"0":0,"1":1},"string":"Hello, 世界"}`,
	tokens: []Token{
		BeginObject,
		String("literals"), BeginArray, Null, False, True, EndArray,
		String("string"), String("Hello, 世界"),
		String("number"), Float(3.14159),
		String("arrayN0"), BeginArray, EndArray,
		String("arrayN1"), BeginArray, Uint(0), EndArray,
		String("arrayN2"), BeginArray, Uint(0), Uint(1), EndArray,
		String("objectN0"), BeginObject, EndObject,
		String("objectN1"), BeginObject, String("0"), Uint(0), EndObject,
		String("objectN2"), BeginObject, String("0"), Uint(0), String("1"), Uint(1), EndObject,
		EndObject,
	},
	pointers: []Pointer{
		"",
		"/literals", "/literals",
		"/literals/0",
		"/literals/1",
		"/literals/2",
		"/literals",
		"/string", "/string",
		"/number", "/number",
		"/arrayN0", "/arrayN0", "/arrayN0",
		"/arrayN1", "/arrayN1",
		"/arrayN1/0",
		"/arrayN1",
		"/arrayN2", "/arrayN2",
		"/arrayN2/0",
		"/arrayN2/1",
		"/arrayN2",
		"/objectN0", "/objectN0", "/objectN0",
		"/objectN1", "/objectN1",
		"/objectN1/0", "/objectN1/0",
		"/objectN1",
		"/objectN2", "/objectN2",
		"/objectN2/0", "/objectN2/0",
		"/objectN2/1", "/objectN2/1",
		"/objectN2",
		"",
	},
}}

// TestCoderInterleaved tests that we can interleave calls that operate on
// tokens and raw values. The only error condition is trying to operate on a
// raw value when the next token is an end of object or array.
func TestCoderInterleaved(t *testing.T) {
	for _, td := range coderTestdata {
		// In TokenFirst and ValueFirst, alternate between tokens and values.
		// In TokenDelims, only use tokens for object and array delimiters.
		for _, modeName := range []string{"TokenFirst", "ValueFirst", "TokenDelims"} {
			t.Run(path.Join(td.name.Name, modeName), func(t *testing.T) {
				testCoderInterleaved(t, td.name.Where, modeName, td)
			})
		}
	}
}
func testCoderInterleaved(t *testing.T, where jsontest.CasePos, modeName string, td coderTestdataEntry) {
	src := strings.NewReader(td.in)
	dst := new(bytes.Buffer)
	dec := NewDecoder(src)
	enc := NewEncoder(dst)
	tickTock := modeName == "TokenFirst"
	for {
		if modeName == "TokenDelims" {
			switch dec.PeekKind() {
			case '{', '}', '[', ']':
				tickTock = true // as token
			default:
				tickTock = false // as value
			}
		}
		if tickTock {
			tok, err := dec.ReadToken()
			if err != nil {
				if err == io.EOF {
					break
				}
				t.Fatalf("%s: Decoder.ReadToken error: %v", where, err)
			}
			if err := enc.WriteToken(tok); err != nil {
				t.Fatalf("%s: Encoder.WriteToken error: %v", where, err)
			}
		} else {
			val, err := dec.ReadValue()
			if err != nil {
				// It is a syntactic error to call ReadValue
				// at the end of an object or array.
				// Retry as a ReadToken call.
				expectError := dec.PeekKind() == '}' || dec.PeekKind() == ']'
				if expectError {
					if !errors.As(err, new(*SyntacticError)) {
						t.Fatalf("%s: Decoder.ReadToken error is %T, want %T", where, err, new(SyntacticError))
					}
					tickTock = !tickTock
					continue
				}

				if err == io.EOF {
					break
				}
				t.Fatalf("%s: Decoder.ReadValue error: %v", where, err)
			}
			if err := enc.WriteValue(val); err != nil {
				t.Fatalf("%s: Encoder.WriteValue error: %v", where, err)
			}
		}
		tickTock = !tickTock
	}

	got := dst.String()
	want := td.outCompacted + "\n"
	if got != want {
		t.Fatalf("%s: output mismatch:\ngot  %q\nwant %q", where, got, want)
	}
}

func TestCoderStackPointer(t *testing.T) {
	tests := []struct {
		token Token
		want  Pointer
	}{
		{Null, ""},

		{BeginArray, ""},
		{EndArray, ""},

		{BeginArray, ""},
		{Bool(true), "/0"},
		{EndArray, ""},

		{BeginArray, ""},
		{String("hello"), "/0"},
		{String("goodbye"), "/1"},
		{EndArray, ""},

		{BeginObject, ""},
		{EndObject, ""},

		{BeginObject, ""},
		{String("hello"), "/hello"},
		{String("goodbye"), "/hello"},
		{EndObject, ""},

		{BeginObject, ""},
		{String(""), "/"},
		{Null, "/"},
		{String("0"), "/0"},
		{Null, "/0"},
		{String("~"), "/~0"},
		{Null, "/~0"},
		{String("/"), "/~1"},
		{Null, "/~1"},
		{String("a//b~/c/~d~~e"), "/a~1~1b~0~1c~1~0d~0~0e"},
		{Null, "/a~1~1b~0~1c~1~0d~0~0e"},
		{String(" \r\n\t"), "/ \r\n\t"},
		{Null, "/ \r\n\t"},
		{EndObject, ""},

		{BeginArray, ""},
		{BeginObject, "/0"},
		{String(""), "/0/"},
		{BeginArray, "/0/"},
		{BeginObject, "/0//0"},
		{String("#"), "/0//0/#"},
		{Null, "/0//0/#"},
		{EndObject, "/0//0"},
		{EndArray, "/0/"},
		{EndObject, "/0"},
		{EndArray, ""},
	}

	for _, allowDupes := range []bool{false, true} {
		var name string
		switch allowDupes {
		case false:
			name = "RejectDuplicateNames"
		case true:
			name = "AllowDuplicateNames"
		}

		t.Run(name, func(t *testing.T) {
			bb := new(bytes.Buffer)

			enc := NewEncoder(bb, AllowDuplicateNames(allowDupes))
			for i, tt := range tests {
				if err := enc.WriteToken(tt.token); err != nil {
					t.Fatalf("%d: Encoder.WriteToken error: %v", i, err)
				}
				if got := enc.StackPointer(); got != tests[i].want {
					t.Fatalf("%d: Encoder.StackPointer = %v, want %v", i, got, tests[i].want)
				}
			}

			dec := NewDecoder(bb, AllowDuplicateNames(allowDupes))
			for i := range tests {
				if _, err := dec.ReadToken(); err != nil {
					t.Fatalf("%d: Decoder.ReadToken error: %v", i, err)
				}
				if got := dec.StackPointer(); got != tests[i].want {
					t.Fatalf("%d: Decoder.StackPointer = %v, want %v", i, got, tests[i].want)
				}
			}
		})
	}
}

func TestCoderMaxDepth(t *testing.T) {
	trimArray := func(b []byte) []byte { return b[len(`[`) : len(b)-len(`]`)] }
	maxArrays := []byte(strings.Repeat(`[`, maxNestingDepth+1) + strings.Repeat(`]`, maxNestingDepth+1))
	trimObject := func(b []byte) []byte { return b[len(`{"":`) : len(b)-len(`}`)] }
	maxObjects := []byte(strings.Repeat(`{"":`, maxNestingDepth+1) + `""` + strings.Repeat(`}`, maxNestingDepth+1))

	t.Run("Decoder", func(t *testing.T) {
		var dec Decoder
		checkReadToken := func(t *testing.T, wantKind Kind, wantErr error) {
			t.Helper()
			if tok, err := dec.ReadToken(); tok.Kind() != wantKind || !equalError(err, wantErr) {
				t.Fatalf("Decoder.ReadToken = (%q, %v), want (%q, %v)", byte(tok.Kind()), err, byte(wantKind), wantErr)
			}
		}
		checkReadValue := func(t *testing.T, wantLen int, wantErr error) {
			t.Helper()
			if val, err := dec.ReadValue(); len(val) != wantLen || !equalError(err, wantErr) {
				t.Fatalf("Decoder.ReadValue = (%d, %v), want (%d, %v)", len(val), err, wantLen, wantErr)
			}
		}

		t.Run("ArraysValid/SingleValue", func(t *testing.T) {
			dec.s.reset(trimArray(maxArrays), nil)
			checkReadValue(t, maxNestingDepth*len(`[]`), nil)
		})
		t.Run("ArraysValid/TokenThenValue", func(t *testing.T) {
			dec.s.reset(trimArray(maxArrays), nil)
			checkReadToken(t, '[', nil)
			checkReadValue(t, (maxNestingDepth-1)*len(`[]`), nil)
			checkReadToken(t, ']', nil)
		})
		t.Run("ArraysValid/AllTokens", func(t *testing.T) {
			dec.s.reset(trimArray(maxArrays), nil)
			for range maxNestingDepth {
				checkReadToken(t, '[', nil)
			}
			for range maxNestingDepth {
				checkReadToken(t, ']', nil)
			}
		})

		wantErr := &SyntacticError{
			ByteOffset:  maxNestingDepth,
			JSONPointer: Pointer(strings.Repeat("/0", maxNestingDepth)),
			Err:         errMaxDepth,
		}
		t.Run("ArraysInvalid/SingleValue", func(t *testing.T) {
			dec.s.reset(maxArrays, nil)
			checkReadValue(t, 0, wantErr)
		})
		t.Run("ArraysInvalid/TokenThenValue", func(t *testing.T) {
			dec.s.reset(maxArrays, nil)
			checkReadToken(t, '[', nil)
			checkReadValue(t, 0, wantErr)
		})
		t.Run("ArraysInvalid/AllTokens", func(t *testing.T) {
			dec.s.reset(maxArrays, nil)
			for range maxNestingDepth {
				checkReadToken(t, '[', nil)
			}
			checkReadValue(t, 0, wantErr)
		})

		t.Run("ObjectsValid/SingleValue", func(t *testing.T) {
			dec.s.reset(trimObject(maxObjects), nil)
			checkReadValue(t, maxNestingDepth*len(`{"":}`)+len(`""`), nil)
		})
		t.Run("ObjectsValid/TokenThenValue", func(t *testing.T) {
			dec.s.reset(trimObject(maxObjects), nil)
			checkReadToken(t, '{', nil)
			checkReadToken(t, '"', nil)
			checkReadValue(t, (maxNestingDepth-1)*len(`{"":}`)+len(`""`), nil)
			checkReadToken(t, '}', nil)
		})
		t.Run("ObjectsValid/AllTokens", func(t *testing.T) {
			dec.s.reset(trimObject(maxObjects), nil)
			for range maxNestingDepth {
				checkReadToken(t, '{', nil)
				checkReadToken(t, '"', nil)
			}
			checkReadToken(t, '"', nil)
			for range maxNestingDepth {
				checkReadToken(t, '}', nil)
			}
		})

		wantErr = &SyntacticError{
			ByteOffset:  maxNestingDepth * int64(len(`{"":`)),
			JSONPointer: Pointer(strings.Repeat("/", maxNestingDepth)),
			Err:         errMaxDepth,
		}
		t.Run("ObjectsInvalid/SingleValue", func(t *testing.T) {
			dec.s.reset(maxObjects, nil)
			checkReadValue(t, 0, wantErr)
		})
		t.Run("ObjectsInvalid/TokenThenValue", func(t *testing.T) {
			dec.s.reset(maxObjects, nil)
			checkReadToken(t, '{', nil)
			checkReadToken(t, '"', nil)
			checkReadValue(t, 0, wantErr)
		})
		t.Run("ObjectsInvalid/AllTokens", func(t *testing.T) {
			dec.s.reset(maxObjects, nil)
			for range maxNestingDepth {
				checkReadToken(t, '{', nil)
				checkReadToken(t, '"', nil)
			}
			checkReadToken(t, 0, wantErr)
		})
	})

	t.Run("Encoder", func(t *testing.T) {
		var enc Encoder
		checkWriteToken := func(t *testing.T, tok Token, wantErr error) {
			t.Helper()
			if err := enc.WriteToken(tok); !equalError(err, wantErr) {
				t.Fatalf("Encoder.WriteToken = %v, want %v", err, wantErr)
			}
		}
		checkWriteValue := func(t *testing.T, val Value, wantErr error) {
			t.Helper()
			if err := enc.WriteValue(val); !equalError(err, wantErr) {
				t.Fatalf("Encoder.WriteValue = %v, want %v", err, wantErr)
			}
		}

		wantErr := &SyntacticError{
			ByteOffset:  maxNestingDepth,
			JSONPointer: Pointer(strings.Repeat("/0", maxNestingDepth)),
			Err:         errMaxDepth,
		}
		t.Run("Arrays/SingleValue", func(t *testing.T) {
			enc.s.reset(enc.s.Buf[:0], nil)
			checkWriteValue(t, maxArrays, wantErr)
			checkWriteValue(t, trimArray(maxArrays), nil)
		})
		t.Run("Arrays/TokenThenValue", func(t *testing.T) {
			enc.s.reset(enc.s.Buf[:0], nil)
			checkWriteToken(t, BeginArray, nil)
			checkWriteValue(t, trimArray(maxArrays), wantErr)
			checkWriteValue(t, trimArray(trimArray(maxArrays)), nil)
			checkWriteToken(t, EndArray, nil)
		})
		t.Run("Arrays/AllTokens", func(t *testing.T) {
			enc.s.reset(enc.s.Buf[:0], nil)
			for range maxNestingDepth {
				checkWriteToken(t, BeginArray, nil)
			}
			checkWriteToken(t, BeginArray, wantErr)
			for range maxNestingDepth {
				checkWriteToken(t, EndArray, nil)
			}
		})

		wantErr = &SyntacticError{
			ByteOffset:  maxNestingDepth * int64(len(`{"":`)),
			JSONPointer: Pointer(strings.Repeat("/", maxNestingDepth)),
			Err:         errMaxDepth,
		}
		t.Run("Objects/SingleValue", func(t *testing.T) {
			enc.s.reset(enc.s.Buf[:0], nil)
			checkWriteValue(t, maxObjects, wantErr)
			checkWriteValue(t, trimObject(maxObjects), nil)
		})
		t.Run("Objects/TokenThenValue", func(t *testing.T) {
			enc.s.reset(enc.s.Buf[:0], nil)
			checkWriteToken(t, BeginObject, nil)
			checkWriteToken(t, String(""), nil)
			checkWriteValue(t, trimObject(maxObjects), wantErr)
			checkWriteValue(t, trimObject(trimObject(maxObjects)), nil)
			checkWriteToken(t, EndObject, nil)
		})
		t.Run("Objects/AllTokens", func(t *testing.T) {
			enc.s.reset(enc.s.Buf[:0], nil)
			for range maxNestingDepth - 1 {
				checkWriteToken(t, BeginObject, nil)
				checkWriteToken(t, String(""), nil)
			}
			checkWriteToken(t, BeginObject, nil)
			checkWriteToken(t, String(""), nil)
			checkWriteToken(t, BeginObject, wantErr)
			checkWriteToken(t, String(""), nil)
			for range maxNestingDepth {
				checkWriteToken(t, EndObject, nil)
			}
		})
	})
}

// FaultyBuffer implements io.Reader and io.Writer.
// It may process fewer bytes than the provided buffer
// and may randomly return an error.
type FaultyBuffer struct {
	B []byte

	// MaxBytes is the maximum number of bytes read/written.
	// A random number of bytes within [0, MaxBytes] are processed.
	// A non-positive value is treated as infinity.
	MaxBytes int

	// MayError specifies whether to randomly provide this error.
	// Even if an error is returned, no bytes are dropped.
	MayError error

	// Rand to use for pseudo-random behavior.
	// If nil, it will be initialized with rand.NewSource(0).
	Rand rand.Source
}

func (p *FaultyBuffer) Read(b []byte) (int, error) {
	b = b[:copy(b[:p.mayTruncate(len(b))], p.B)]
	p.B = p.B[len(b):]
	if len(p.B) == 0 && (len(b) == 0 || p.randN(2) == 0) {
		return len(b), io.EOF
	}
	return len(b), p.mayError()
}

func (p *FaultyBuffer) Write(b []byte) (int, error) {
	b2 := b[:p.mayTruncate(len(b))]
	p.B = append(p.B, b2...)
	if len(b2) < len(b) {
		return len(b2), io.ErrShortWrite
	}
	return len(b2), p.mayError()
}

// mayTruncate may return a value between [0, n].
func (p *FaultyBuffer) mayTruncate(n int) int {
	if p.MaxBytes > 0 {
		if n > p.MaxBytes {
			n = p.MaxBytes
		}
		return p.randN(n + 1)
	}
	return n
}

// mayError may return a non-nil error.
func (p *FaultyBuffer) mayError() error {
	if p.MayError != nil && p.randN(2) == 0 {
		return p.MayError
	}
	return nil
}

func (p *FaultyBuffer) randN(n int) int {
	if p.Rand == nil {
		p.Rand = rand.NewSource(0)
	}
	return int(p.Rand.Int63() % int64(n))
}
