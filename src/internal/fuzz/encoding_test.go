// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fuzz

import (
	"math"
	"strconv"
	"testing"
	"unicode"
)

func TestUnmarshalMarshal(t *testing.T) {
	var tests = []struct {
		desc   string
		in     string
		reject bool
		want   string // if different from in
	}{
		{
			desc:   "missing version",
			in:     "int(1234)",
			reject: true,
		},
		{
			desc: "malformed string",
			in: `go test fuzz v1
string("a"bcad")`,
			reject: true,
		},
		{
			desc: "empty value",
			in: `go test fuzz v1
int()`,
			reject: true,
		},
		{
			desc: "negative uint",
			in: `go test fuzz v1
uint(-32)`,
			reject: true,
		},
		{
			desc: "int8 too large",
			in: `go test fuzz v1
int8(1234456)`,
			reject: true,
		},
		{
			desc: "multiplication in int value",
			in: `go test fuzz v1
int(20*5)`,
			reject: true,
		},
		{
			desc: "double negation",
			in: `go test fuzz v1
int(--5)`,
			reject: true,
		},
		{
			desc: "malformed bool",
			in: `go test fuzz v1
bool(0)`,
			reject: true,
		},
		{
			desc: "malformed byte",
			in: `go test fuzz v1
byte('aa)`,
			reject: true,
		},
		{
			desc: "byte out of range",
			in: `go test fuzz v1
byte('☃')`,
			reject: true,
		},
		{
			desc: "extra newline",
			in: `go test fuzz v1
string("has extra newline")
`,
			want: `go test fuzz v1
string("has extra newline")`,
		},
		{
			desc: "trailing spaces",
			in: `go test fuzz v1
string("extra")
[]byte("spacing")  
    `,
			want: `go test fuzz v1
string("extra")
[]byte("spacing")`,
		},
		{
			desc: "float types",
			in: `go test fuzz v1
float64(0)
float32(0)`,
		},
		{
			desc: "various types",
			in: `go test fuzz v1
int(-23)
int8(-2)
int64(2342425)
uint(1)
uint16(234)
uint32(352342)
uint64(123)
rune('œ')
byte('K')
byte('ÿ')
[]byte("hello¿")
[]byte("a")
bool(true)
string("hello\\xbd\\xb2=\\xbc ⌘")
float64(-12.5)
float32(2.5)`,
		},
		{
			desc: "float edge cases",
			// The two IEEE 754 bit patterns used for the math.Float{64,32}frombits
			// encodings are non-math.NAN quiet-NaN values. Since they are not equal
			// to math.NaN(), they should be re-encoded to their bit patterns. They
			// are, respectively:
			//   * math.Float64bits(math.NaN())+1
			//   * math.Float32bits(float32(math.NaN()))+1
			in: `go test fuzz v1
float32(-0)
float64(-0)
float32(+Inf)
float32(-Inf)
float32(NaN)
float64(+Inf)
float64(-Inf)
float64(NaN)
math.Float64frombits(0x7ff8000000000002)
math.Float32frombits(0x7fc00001)`,
		},
		{
			desc: "int variations",
			// Although we arbitrarily choose default integer bases (0 or 16), we may
			// want to change those arbitrary choices in the future and should not
			// break the parser. Verify that integers in the opposite bases still
			// parse correctly.
			in: `go test fuzz v1
int(0x0)
int32(0x41)
int64(0xfffffffff)
uint32(0xcafef00d)
uint64(0xffffffffffffffff)
uint8(0b0000000)
byte(0x0)
byte('\000')
byte('\u0000')
byte('\'')
math.Float64frombits(9221120237041090562)
math.Float32frombits(2143289345)`,
			want: `go test fuzz v1
int(0)
rune('A')
int64(68719476735)
uint32(3405705229)
uint64(18446744073709551615)
byte('\x00')
byte('\x00')
byte('\x00')
byte('\x00')
byte('\'')
math.Float64frombits(0x7ff8000000000002)
math.Float32frombits(0x7fc00001)`,
		},
		{
			desc: "rune validation",
			in: `go test fuzz v1
rune(0)
rune(0x41)
rune(-1)
rune(0xfffd)
rune(0xd800)
rune(0x10ffff)
rune(0x110000)
`,
			want: `go test fuzz v1
rune('\x00')
rune('A')
int32(-1)
rune('�')
int32(55296)
rune('\U0010ffff')
int32(1114112)`,
		},
		{
			desc: "int overflow",
			in: `go test fuzz v1
int(0x7fffffffffffffff)
uint(0xffffffffffffffff)`,
			want: func() string {
				switch strconv.IntSize {
				case 32:
					return `go test fuzz v1
int(-1)
uint(4294967295)`
				case 64:
					return `go test fuzz v1
int(9223372036854775807)
uint(18446744073709551615)`
				default:
					panic("unreachable")
				}
			}(),
		},
	}
	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			vals, err := unmarshalCorpusFile([]byte(test.in))
			if test.reject {
				if err == nil {
					t.Fatalf("unmarshal unexpected success")
				}
				return
			}
			if err != nil {
				t.Fatalf("unmarshal unexpected error: %v", err)
			}
			newB := marshalCorpusFile(vals...)
			if err != nil {
				t.Fatalf("marshal unexpected error: %v", err)
			}
			if newB[len(newB)-1] != '\n' {
				t.Error("didn't write final newline to corpus file")
			}

			want := test.want
			if want == "" {
				want = test.in
			}
			want += "\n"
			got := string(newB)
			if got != want {
				t.Errorf("unexpected marshaled value\ngot:\n%s\nwant:\n%s", got, want)
			}
		})
	}
}

// BenchmarkMarshalCorpusFile measures the time it takes to serialize byte
// slices of various sizes to a corpus file. The slice contains a repeating
// sequence of bytes 0-255 to mix escaped and non-escaped characters.
func BenchmarkMarshalCorpusFile(b *testing.B) {
	buf := make([]byte, 1024*1024)
	for i := 0; i < len(buf); i++ {
		buf[i] = byte(i)
	}

	for sz := 1; sz <= len(buf); sz <<= 1 {
		sz := sz
		b.Run(strconv.Itoa(sz), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				b.SetBytes(int64(sz))
				marshalCorpusFile(buf[:sz])
			}
		})
	}
}

// BenchmarkUnmarshalCorpusfile measures the time it takes to deserialize
// files encoding byte slices of various sizes. The slice contains a repeating
// sequence of bytes 0-255 to mix escaped and non-escaped characters.
func BenchmarkUnmarshalCorpusFile(b *testing.B) {
	buf := make([]byte, 1024*1024)
	for i := 0; i < len(buf); i++ {
		buf[i] = byte(i)
	}

	for sz := 1; sz <= len(buf); sz <<= 1 {
		sz := sz
		data := marshalCorpusFile(buf[:sz])
		b.Run(strconv.Itoa(sz), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				b.SetBytes(int64(sz))
				unmarshalCorpusFile(data)
			}
		})
	}
}

func TestByteRoundTrip(t *testing.T) {
	for x := 0; x < 256; x++ {
		b1 := byte(x)
		buf := marshalCorpusFile(b1)
		vs, err := unmarshalCorpusFile(buf)
		if err != nil {
			t.Fatal(err)
		}
		b2 := vs[0].(byte)
		if b2 != b1 {
			t.Fatalf("unmarshaled %v, want %v:\n%s", b2, b1, buf)
		}
	}
}

func TestInt8RoundTrip(t *testing.T) {
	for x := -128; x < 128; x++ {
		i1 := int8(x)
		buf := marshalCorpusFile(i1)
		vs, err := unmarshalCorpusFile(buf)
		if err != nil {
			t.Fatal(err)
		}
		i2 := vs[0].(int8)
		if i2 != i1 {
			t.Fatalf("unmarshaled %v, want %v:\n%s", i2, i1, buf)
		}
	}
}

func FuzzFloat64RoundTrip(f *testing.F) {
	f.Add(math.Float64bits(0))
	f.Add(math.Float64bits(math.Copysign(0, -1)))
	f.Add(math.Float64bits(math.MaxFloat64))
	f.Add(math.Float64bits(math.SmallestNonzeroFloat64))
	f.Add(math.Float64bits(math.NaN()))
	f.Add(uint64(0x7FF0000000000001)) // signaling NaN
	f.Add(math.Float64bits(math.Inf(1)))
	f.Add(math.Float64bits(math.Inf(-1)))

	f.Fuzz(func(t *testing.T, u1 uint64) {
		x1 := math.Float64frombits(u1)

		b := marshalCorpusFile(x1)
		t.Logf("marshaled math.Float64frombits(0x%x):\n%s", u1, b)

		xs, err := unmarshalCorpusFile(b)
		if err != nil {
			t.Fatal(err)
		}
		if len(xs) != 1 {
			t.Fatalf("unmarshaled %d values", len(xs))
		}
		x2 := xs[0].(float64)
		u2 := math.Float64bits(x2)
		if u2 != u1 {
			t.Errorf("unmarshaled %v (bits 0x%x)", x2, u2)
		}
	})
}

func FuzzRuneRoundTrip(f *testing.F) {
	f.Add(rune(-1))
	f.Add(rune(0xd800))
	f.Add(rune(0xdfff))
	f.Add(rune(unicode.ReplacementChar))
	f.Add(rune(unicode.MaxASCII))
	f.Add(rune(unicode.MaxLatin1))
	f.Add(rune(unicode.MaxRune))
	f.Add(rune(unicode.MaxRune + 1))
	f.Add(rune(-0x80000000))
	f.Add(rune(0x7fffffff))

	f.Fuzz(func(t *testing.T, r1 rune) {
		b := marshalCorpusFile(r1)
		t.Logf("marshaled rune(0x%x):\n%s", r1, b)

		rs, err := unmarshalCorpusFile(b)
		if err != nil {
			t.Fatal(err)
		}
		if len(rs) != 1 {
			t.Fatalf("unmarshaled %d values", len(rs))
		}
		r2 := rs[0].(rune)
		if r2 != r1 {
			t.Errorf("unmarshaled rune(0x%x)", r2)
		}
	})
}

func FuzzStringRoundTrip(f *testing.F) {
	f.Add("")
	f.Add("\x00")
	f.Add(string([]rune{unicode.ReplacementChar}))

	f.Fuzz(func(t *testing.T, s1 string) {
		b := marshalCorpusFile(s1)
		t.Logf("marshaled %q:\n%s", s1, b)

		rs, err := unmarshalCorpusFile(b)
		if err != nil {
			t.Fatal(err)
		}
		if len(rs) != 1 {
			t.Fatalf("unmarshaled %d values", len(rs))
		}
		s2 := rs[0].(string)
		if s2 != s1 {
			t.Errorf("unmarshaled %q", s2)
		}
	})
}
