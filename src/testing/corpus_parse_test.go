// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testing

import (
	"math"
	"reflect"
)

func TestParseCorpusValue(t *T) {
	tests := []struct {
		name    string
		input   string
		want    any
		wantErr bool
	}{
		// String types
		{name: "string simple", input: `string("hello")`, want: "hello"},
		{name: "string empty", input: `string("")`, want: ""},
		{name: "string with spaces", input: `string("hello world")`, want: "hello world"},
		{name: "string with escapes", input: `string("hello\nworld")`, want: "hello\nworld"},
		{name: "string with quotes", input: `string("say \"hi\"")`, want: `say "hi"`},
		{
			name:  "string with unicode",
			input: `string("こんにちは")`,
			want:  "こんにちは",
		},

		// []byte types
		{name: "bytes simple", input: `[]byte("hello")`, want: []byte("hello")},
		{name: "bytes empty", input: `[]byte("")`, want: []byte("")},
		{name: "bytes with null", input: `[]byte("a\x00b")`, want: []byte("a\x00b")},
		{name: "bytes with escapes", input: `[]byte("\x01\x02\x03")`, want: []byte{1, 2, 3}},

		// Boolean types
		{name: "bool true", input: `bool(true)`, want: true},
		{name: "bool false", input: `bool(false)`, want: false},

		// Signed integer types
		{name: "int positive", input: `int(42)`, want: int(42)},
		{name: "int negative", input: `int(-42)`, want: int(-42)},
		{name: "int zero", input: `int(0)`, want: int(0)},
		{name: "int8 positive", input: `int8(127)`, want: int8(127)},
		{name: "int8 negative", input: `int8(-128)`, want: int8(-128)},
		{name: "int16 positive", input: `int16(32767)`, want: int16(32767)},
		{name: "int16 negative", input: `int16(-32768)`, want: int16(-32768)},
		{name: "int32 positive", input: `int32(2147483647)`, want: int32(2147483647)},
		{name: "int32 negative", input: `int32(-2147483648)`, want: int32(-2147483648)},
		{
			name:  "int64 positive",
			input: `int64(9223372036854775807)`,
			want:  int64(9223372036854775807),
		},
		{
			name:  "int64 negative",
			input: `int64(-9223372036854775808)`,
			want:  int64(-9223372036854775808),
		},

		// Unsigned integer types
		{name: "uint positive", input: `uint(42)`, want: uint(42)},
		{name: "uint zero", input: `uint(0)`, want: uint(0)},
		{name: "uint8 max", input: `uint8(255)`, want: uint8(255)},
		{name: "uint16 max", input: `uint16(65535)`, want: uint16(65535)},
		{name: "uint32 max", input: `uint32(4294967295)`, want: uint32(4294967295)},
		{
			name:  "uint64 max",
			input: `uint64(18446744073709551615)`,
			want:  uint64(18446744073709551615),
		},

		// Hex literals
		{name: "int hex", input: `int(0x2A)`, want: int(42)},
		{name: "uint hex", input: `uint(0xFF)`, want: uint(255)},

		// Byte as character
		{name: "byte char", input: `byte('A')`, want: byte('A')},
		{name: "byte escape", input: `byte('\n')`, want: byte('\n')},
		{name: "byte hex", input: `byte('\x41')`, want: byte('A')},
		{name: "byte int", input: `byte(65)`, want: uint8(65)},

		// Rune types
		{name: "rune char", input: `rune('A')`, want: rune('A')},
		{name: "rune unicode", input: `rune('★')`, want: rune('★')},
		{name: "rune escape", input: `rune('\n')`, want: rune('\n')},
		{name: "rune int", input: `rune(9733)`, want: int32(9733)}, // ★ = U+2605 = 9733

		// Float types
		{name: "float32 positive", input: `float32(3.14)`, want: float32(3.14)},
		{name: "float32 negative", input: `float32(-3.14)`, want: float32(-3.14)},
		{name: "float32 int", input: `float32(42)`, want: float32(42)},
		{
			name:  "float64 positive",
			input: `float64(3.14159265358979)`,
			want:  float64(3.14159265358979),
		},
		{
			name:  "float64 negative",
			input: `float64(-3.14159265358979)`,
			want:  float64(-3.14159265358979),
		},
		{name: "float64 int", input: `float64(42)`, want: float64(42)},

		// Special float values
		{name: "float64 NaN", input: `float64(NaN)`, want: math.NaN()},
		{name: "float64 +Inf", input: `float64(+Inf)`, want: math.Inf(1)},
		{name: "float64 -Inf", input: `float64(-Inf)`, want: math.Inf(-1)},
		{name: "float32 NaN", input: `float32(NaN)`, want: float32(math.NaN())},
		{name: "float32 +Inf", input: `float32(+Inf)`, want: float32(math.Inf(1))},
		{name: "float32 -Inf", input: `float32(-Inf)`, want: float32(math.Inf(-1))},

		// math.Float*frombits (3.14)
		{
			name:  "float64 frombits",
			input: `math.Float64frombits(4614256656552045848)`,
			want:  math.Float64frombits(4614256656552045848),
		},
		{
			name:  "float32 frombits",
			input: `math.Float32frombits(1078523331)`,
			want:  math.Float32frombits(1078523331),
		},

		// Error cases
		{name: "invalid type", input: `invalid("test")`, wantErr: true},
		{name: "wrong bool value", input: `bool(maybe)`, wantErr: true},
		{name: "missing parens", input: `string"hello"`, wantErr: true},
		{name: "wrong arg count", input: `string("a", "b")`, wantErr: true},
		{name: "empty", input: ``, wantErr: true},
		{name: "int overflow", input: `int8(200)`, wantErr: true},
		{name: "negative uint", input: `uint(-1)`, wantErr: true},
		{name: "byte too large", input: `byte('日')`, wantErr: true}, // Multi-byte rune
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *T) {
			got, err := parseCorpusValue([]byte(tt.input))
			if (err != nil) != tt.wantErr {
				t.Errorf("parseCorpusValue(%q) error = %v, wantErr %v", tt.input, err, tt.wantErr)
				return
			}
			if tt.wantErr {
				return
			}

			// Special handling for NaN since NaN != NaN
			if f, ok := tt.want.(float64); ok && math.IsNaN(f) {
				if g, ok := got.(float64); !ok || !math.IsNaN(g) {
					t.Errorf("parseCorpusValue(%q) = %v, want NaN", tt.input, got)
				}
				return
			}
			if f, ok := tt.want.(float32); ok && math.IsNaN(float64(f)) {
				if g, ok := got.(float32); !ok || !math.IsNaN(float64(g)) {
					t.Errorf("parseCorpusValue(%q) = %v, want NaN", tt.input, got)
				}
				return
			}

			// For []byte, use bytes comparison
			if b, ok := tt.want.([]byte); ok {
				if g, ok := got.([]byte); !ok || string(g) != string(b) {
					t.Errorf("parseCorpusValue(%q) = %v, want %v", tt.input, got, tt.want)
				}
				return
			}

			if got != tt.want {
				t.Errorf("parseCorpusValue(%q) = %v (%T), want %v (%T)",
					tt.input, got, got, tt.want, tt.want)
			}
		})
	}
}

func TestParseGoCorpusFile(t *T) {
	tests := []struct {
		name    string
		data    string
		types   []reflect.Type
		want    []any
		wantErr bool
	}{
		{
			name:  "single string",
			data:  "go test fuzz v1\nstring(\"hello\")\n",
			types: []reflect.Type{reflect.TypeOf("")},
			want:  []any{"hello"},
		},
		{
			name: "string and int",
			data: "go test fuzz v1\nstring(\"test\")\nint(42)\n",
			types: []reflect.Type{
				reflect.TypeOf(""),
				reflect.TypeOf(int(0)),
			},
			want: []any{"test", int(42)},
		},
		{
			name: "bytes and bool",
			data: "go test fuzz v1\n[]byte(\"data\")\nbool(true)\n",
			types: []reflect.Type{
				reflect.TypeOf([]byte{}),
				reflect.TypeOf(true),
			},
			want: []any{[]byte("data"), true},
		},
		{
			name: "multiple ints",
			data: "go test fuzz v1\nint8(-10)\nint16(1000)\nint32(100000)\n" +
				"int64(10000000000)\n",
			types: []reflect.Type{
				reflect.TypeOf(int8(0)),
				reflect.TypeOf(int16(0)),
				reflect.TypeOf(int32(0)),
				reflect.TypeOf(int64(0)),
			},
			want: []any{int8(-10), int16(1000), int32(100000), int64(10000000000)},
		},
		{
			name: "with blank lines",
			data: "go test fuzz v1\n\nstring(\"a\")\n\nint(1)\n\n",
			types: []reflect.Type{
				reflect.TypeOf(""),
				reflect.TypeOf(int(0)),
			},
			want: []any{"a", int(1)},
		},
		{
			name:  "with CRLF line endings",
			data:  "go test fuzz v1\r\nstring(\"test\")\r\n",
			types: []reflect.Type{reflect.TypeOf("")},
			want:  []any{"test"},
		},

		// Error cases
		{
			name:    "empty file",
			data:    "",
			types:   []reflect.Type{reflect.TypeOf("")},
			wantErr: true,
		},
		{
			name:    "missing version",
			data:    "string(\"hello\")\n",
			types:   []reflect.Type{reflect.TypeOf("")},
			wantErr: true,
		},
		{
			name:    "wrong version",
			data:    "go test fuzz v2\nstring(\"hello\")\n",
			types:   []reflect.Type{reflect.TypeOf("")},
			wantErr: true,
		},
		{
			name:    "type mismatch",
			data:    "go test fuzz v1\nstring(\"hello\")\n",
			types:   []reflect.Type{reflect.TypeOf(int(0))},
			wantErr: true,
		},
		{
			name: "wrong number of values - too few",
			data: "go test fuzz v1\nstring(\"hello\")\n",
			types: []reflect.Type{
				reflect.TypeOf(""),
				reflect.TypeOf(int(0)),
			},
			wantErr: true,
		},
		{
			name:    "wrong number of values - too many",
			data:    "go test fuzz v1\nstring(\"hello\")\nint(42)\n",
			types:   []reflect.Type{reflect.TypeOf("")},
			wantErr: true,
		},
		{
			name:    "malformed value",
			data:    "go test fuzz v1\nstring(hello)\n", // missing quotes
			types:   []reflect.Type{reflect.TypeOf("")},
			wantErr: true,
		},
		{
			name:    "version only",
			data:    "go test fuzz v1\n",
			types:   []reflect.Type{reflect.TypeOf("")},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *T) {
			got, err := parseGoCorpusFile([]byte(tt.data), tt.types)
			if (err != nil) != tt.wantErr {
				t.Errorf("parseGoCorpusFile() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if tt.wantErr {
				return
			}

			if len(got) != len(tt.want) {
				t.Errorf("parseGoCorpusFile() got %d values, want %d", len(got), len(tt.want))
				return
			}

			for i := range got {
				// Handle []byte specially
				if b, ok := tt.want[i].([]byte); ok {
					if g, ok := got[i].([]byte); !ok || string(g) != string(b) {
						t.Errorf("parseGoCorpusFile() value[%d] = %v, want %v", i, got[i], tt.want[i])
					}
					continue
				}
				if got[i] != tt.want[i] {
					t.Errorf("parseGoCorpusFile() value[%d] = %v (%T), want %v (%T)",
						i, got[i], got[i], tt.want[i], tt.want[i])
				}
			}
		})
	}
}

func TestParseCharLiteral(t *T) {
	tests := []struct {
		name    string
		input   string
		want    rune
		wantErr bool
	}{
		{name: "simple char", input: "'A'", want: 'A'},
		{name: "unicode char", input: "'★'", want: '★'},
		{name: "newline escape", input: `'\n'`, want: '\n'},
		{name: "tab escape", input: `'\t'`, want: '\t'},
		{name: "hex escape", input: `'\x41'`, want: 'A'},
		{name: "unicode escape", input: `'\u2605'`, want: '★'},
		{name: "single quote escape", input: `'\''`, want: '\''},
		{name: "backslash escape", input: `'\\'`, want: '\\'},

		// Error cases
		{name: "empty", input: "''", wantErr: true},
		{name: "missing quotes", input: "A", wantErr: true},
		{name: "missing end quote", input: "'A", wantErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *T) {
			got, err := parseCharLiteral(tt.input)
			if (err != nil) != tt.wantErr {
				t.Errorf("parseCharLiteral(%q) error = %v, wantErr %v", tt.input, err, tt.wantErr)
				return
			}
			if !tt.wantErr && got != tt.want {
				t.Errorf("parseCharLiteral(%q) = %v, want %v", tt.input, got, tt.want)
			}
		})
	}
}

func TestMarshalGoCorpusFile(t *T) {
	tests := []struct {
		name string
		vals []any
		want string
	}{
		{
			name: "single string",
			vals: []any{"hello"},
			want: "go test fuzz v1\nstring(\"hello\")\n",
		},
		{
			name: "single bytes",
			vals: []any{[]byte("world")},
			want: "go test fuzz v1\n[]byte(\"world\")\n",
		},
		{
			name: "bool true",
			vals: []any{true},
			want: "go test fuzz v1\nbool(true)\n",
		},
		{
			name: "bool false",
			vals: []any{false},
			want: "go test fuzz v1\nbool(false)\n",
		},
		{
			name: "integers",
			vals: []any{int(42), int8(-1), int16(1000), int64(9999999999)},
			want: "go test fuzz v1\nint(42)\nint8(-1)\nint16(1000)\nint64(9999999999)\n",
		},
		{
			name: "unsigned integers",
			vals: []any{uint(42), uint16(1000), uint32(100000), uint64(9999999999)},
			want: "go test fuzz v1\nuint(42)\nuint16(1000)\nuint32(100000)\nuint64(9999999999)\n",
		},
		{
			name: "byte as char",
			vals: []any{byte('A')},
			want: "go test fuzz v1\nbyte('A')\n",
		},
		{
			name: "rune",
			vals: []any{rune('★')},
			want: "go test fuzz v1\nrune('★')\n",
		},
		{
			name: "negative int32 as int32 not rune",
			vals: []any{int32(-1)},
			want: "go test fuzz v1\nint32(-1)\n",
		},
		{
			name: "float32",
			vals: []any{float32(3.14)},
			want: "go test fuzz v1\nfloat32(3.14)\n",
		},
		{
			name: "float64",
			vals: []any{float64(3.14159)},
			want: "go test fuzz v1\nfloat64(3.14159)\n",
		},
		{
			name: "float64 positive infinity",
			vals: []any{math.Inf(1)},
			want: "go test fuzz v1\nfloat64(+Inf)\n",
		},
		{
			name: "float64 negative infinity",
			vals: []any{math.Inf(-1)},
			want: "go test fuzz v1\nfloat64(-Inf)\n",
		},
		{
			name: "float32 positive infinity",
			vals: []any{float32(math.Inf(1))},
			want: "go test fuzz v1\nfloat32(+Inf)\n",
		},
		{
			name: "string with escapes",
			vals: []any{"hello\nworld\t\"quoted\""},
			want: "go test fuzz v1\nstring(\"hello\\nworld\\t\\\"quoted\\\"\")\n",
		},
		{
			name: "multiple types",
			vals: []any{"test", int(42), true, []byte{1, 2, 3}},
			want: "go test fuzz v1\nstring(\"test\")\nint(42)\nbool(true)\n" +
				"[]byte(\"\\x01\\x02\\x03\")\n",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *T) {
			got := marshalGoCorpusFile(tt.vals)
			if string(got) != tt.want {
				t.Errorf("marshalGoCorpusFile(%v) =\n%s\nwant:\n%s", tt.vals, got, tt.want)
			}
		})
	}
}

func TestMarshalGoCorpusFileNaN(t *T) {
	// Test NaN separately since we need special comparison
	vals := []any{math.NaN()}
	got := string(marshalGoCorpusFile(vals))
	want := "go test fuzz v1\nfloat64(NaN)\n"
	if got != want {
		t.Errorf("marshalGoCorpusFile(NaN) = %q, want %q", got, want)
	}

	// Test float32 NaN
	vals = []any{float32(math.NaN())}
	got = string(marshalGoCorpusFile(vals))
	want = "go test fuzz v1\nfloat32(NaN)\n"
	if got != want {
		t.Errorf("marshalGoCorpusFile(float32 NaN) = %q, want %q", got, want)
	}
}

func TestMarshalParseRoundTrip(t *T) {
	// Test that marshaling and parsing produces the same values
	testCases := [][]any{
		{"hello world"},
		{[]byte("binary\x00data")},
		{true},
		{false},
		{int(42)},
		{int8(-128)},
		{int16(32767)},
		{int64(-9223372036854775808)},
		{uint(42)},
		{uint8(255)},
		{uint16(65535)},
		{uint32(4294967295)},
		{uint64(18446744073709551615)},
		{float32(3.14)},
		{float64(2.71828)},
		{math.Inf(1)},
		{math.Inf(-1)},
		{float32(math.Inf(1))},
		{float32(math.Inf(-1))},
		{"multi", int(1), true, []byte("args")},
	}

	for i, vals := range testCases {
		t.Run(string(rune('0'+i)), func(t *T) {
			// Get types
			types := make([]reflect.Type, len(vals))
			for j, v := range vals {
				types[j] = reflect.TypeOf(v)
			}

			// Marshal
			data := marshalGoCorpusFile(vals)

			// Parse
			parsed, err := parseGoCorpusFile(data, types)
			if err != nil {
				t.Fatalf("parseGoCorpusFile failed: %v", err)
			}

			// Compare
			if len(parsed) != len(vals) {
				t.Fatalf("got %d values, want %d", len(parsed), len(vals))
			}

			for j := range vals {
				// Special handling for []byte
				if b, ok := vals[j].([]byte); ok {
					if pb, ok := parsed[j].([]byte); !ok || string(pb) != string(b) {
						t.Errorf("value[%d] = %v, want %v", j, parsed[j], vals[j])
					}
					continue
				}
				// Special handling for NaN
				if f, ok := vals[j].(float64); ok && math.IsNaN(f) {
					if pf, ok := parsed[j].(float64); !ok || !math.IsNaN(pf) {
						t.Errorf("value[%d] = %v, want NaN", j, parsed[j])
					}
					continue
				}
				if f, ok := vals[j].(float32); ok && math.IsNaN(float64(f)) {
					if pf, ok := parsed[j].(float32); !ok || !math.IsNaN(float64(pf)) {
						t.Errorf("value[%d] = %v, want NaN", j, parsed[j])
					}
					continue
				}
				if parsed[j] != vals[j] {
					t.Errorf("value[%d] = %v (%T), want %v (%T)",
						j, parsed[j], parsed[j], vals[j], vals[j])
				}
			}
		})
	}
}
