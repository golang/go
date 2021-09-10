// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fuzz

import (
	"strconv"
	"strings"
	"testing"
)

func TestUnmarshalMarshal(t *testing.T) {
	var tests = []struct {
		in string
		ok bool
	}{
		{
			in: "int(1234)",
			ok: false, // missing version
		},
		{
			in: `go test fuzz v1
string("a"bcad")`,
			ok: false, // malformed
		},
		{
			in: `go test fuzz v1
int()`,
			ok: false, // empty value
		},
		{
			in: `go test fuzz v1
uint(-32)`,
			ok: false, // invalid negative uint
		},
		{
			in: `go test fuzz v1
int8(1234456)`,
			ok: false, // int8 too large
		},
		{
			in: `go test fuzz v1
int(20*5)`,
			ok: false, // expression in int value
		},
		{
			in: `go test fuzz v1
int(--5)`,
			ok: false, // expression in int value
		},
		{
			in: `go test fuzz v1
bool(0)`,
			ok: false, // malformed bool
		},
		{
			in: `go test fuzz v1
byte('aa)`,
			ok: false, // malformed byte
		},
		{
			in: `go test fuzz v1
byte('☃')`,
			ok: false, // byte out of range
		},
		{
			in: `go test fuzz v1
string("has final newline")
`,
			ok: true, // has final newline
		},
		{
			in: `go test fuzz v1
string("extra")
[]byte("spacing")  
    `,
			ok: true, // extra spaces in the final newline
		},
		{
			in: `go test fuzz v1
float64(0)
float32(0)`,
			ok: true, // will be an integer literal since there is no decimal
		},
		{
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
			ok: true,
		},
	}
	for _, test := range tests {
		t.Run(test.in, func(t *testing.T) {
			vals, err := unmarshalCorpusFile([]byte(test.in))
			if test.ok && err != nil {
				t.Fatalf("unmarshal unexpected error: %v", err)
			} else if !test.ok && err == nil {
				t.Fatalf("unmarshal unexpected success")
			}
			if !test.ok {
				return // skip the rest of the test
			}
			newB := marshalCorpusFile(vals...)
			if err != nil {
				t.Fatalf("marshal unexpected error: %v", err)
			}
			if newB[len(newB)-1] != '\n' {
				t.Error("didn't write final newline to corpus file")
			}
			before, after := strings.TrimSpace(test.in), strings.TrimSpace(string(newB))
			if before != after {
				t.Errorf("values changed after unmarshal then marshal\nbefore: %q\nafter:  %q", before, after)
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
