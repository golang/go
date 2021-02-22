// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fuzz

import (
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
string("extra")
[]byte("spacing")  
    `,
			ok: true,
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
			want := strings.TrimSpace(test.in)
			if want != string(newB) {
				t.Errorf("values changed after unmarshal then marshal\nbefore: %q\nafter:  %q", want, newB)
			}
		})
	}
}
