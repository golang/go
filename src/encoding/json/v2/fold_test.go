// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jsonv2

package json

import (
	"fmt"
	"reflect"
	"testing"
	"unicode"
)

var equalFoldTestdata = []struct {
	in1, in2 string
	want     bool
}{
	{"", "", true},
	{"abc", "abc", true},
	{"ABcd", "ABcd", true},
	{"123abc", "123ABC", true},
	{"_1_2_-_3__--a-_-b-c-", "123ABC", true},
	{"αβδ", "ΑΒΔ", true},
	{"abc", "xyz", false},
	{"abc", "XYZ", false},
	{"abcdefghijk", "abcdefghijX", false},
	{"abcdefghijk", "abcdefghij\u212A", true},
	{"abcdefghijK", "abcdefghij\u212A", true},
	{"abcdefghijkz", "abcdefghij\u212Ay", false},
	{"abcdefghijKz", "abcdefghij\u212Ay", false},
	{"1", "2", false},
	{"utf-8", "US-ASCII", false},
	{"hello, world!", "hello, world!", true},
	{"hello, world!", "Hello, World!", true},
	{"hello, world!", "HELLO, WORLD!", true},
	{"hello, world!", "jello, world!", false},
	{"γειά, κόσμε!", "γειά, κόσμε!", true},
	{"γειά, κόσμε!", "Γειά, Κόσμε!", true},
	{"γειά, κόσμε!", "ΓΕΙΆ, ΚΌΣΜΕ!", true},
	{"γειά, κόσμε!", "ΛΕΙΆ, ΚΌΣΜΕ!", false},
	{"AESKey", "aesKey", true},
	{"γειά, κόσμε!", "Γ\xce_\xb5ιά, Κόσμε!", false},
	{"aeskey", "AESKEY", true},
	{"AESKEY", "aes_key", true},
	{"aes_key", "AES_KEY", true},
	{"AES_KEY", "aes-key", true},
	{"aes-key", "AES-KEY", true},
	{"AES-KEY", "aesKey", true},
	{"aesKey", "AesKey", true},
	{"AesKey", "AESKey", true},
	{"AESKey", "aeskey", true},
	{"DESKey", "aeskey", false},
	{"AES Key", "aeskey", false},
	{"aes﹏key", "aeskey", false}, // Unicode underscore not handled
	{"aes〰key", "aeskey", false}, // Unicode dash not handled
}

func TestEqualFold(t *testing.T) {
	for _, tt := range equalFoldTestdata {
		got := equalFold([]byte(tt.in1), []byte(tt.in2))
		if got != tt.want {
			t.Errorf("equalFold(%q, %q) = %v, want %v", tt.in1, tt.in2, got, tt.want)
		}
	}
}

func equalFold(x, y []byte) bool {
	return string(foldName(x)) == string(foldName(y))
}

func TestFoldRune(t *testing.T) {
	if testing.Short() {
		t.Skip()
	}

	var foldSet []rune
	for r := range rune(unicode.MaxRune + 1) {
		// Derive all runes that are all part of the same fold set.
		foldSet = foldSet[:0]
		for r0 := r; r != r0 || len(foldSet) == 0; r = unicode.SimpleFold(r) {
			foldSet = append(foldSet, r)
		}

		// Normalized form of each rune in a foldset must be the same and
		// also be within the set itself.
		var withinSet bool
		rr0 := foldRune(foldSet[0])
		for _, r := range foldSet {
			withinSet = withinSet || rr0 == r
			rr := foldRune(r)
			if rr0 != rr {
				t.Errorf("foldRune(%q) = %q, want %q", r, rr, rr0)
			}
		}
		if !withinSet {
			t.Errorf("foldRune(%q) = %q not in fold set %q", foldSet[0], rr0, string(foldSet))
		}
	}
}

// TestBenchmarkUnmarshalUnknown unmarshals an unknown field into a struct with
// varying number of fields. Since the unknown field does not directly match
// any known field by name, it must fall back on case-insensitive matching.
func TestBenchmarkUnmarshalUnknown(t *testing.T) {
	in := []byte(`{"NameUnknown":null}`)
	for _, n := range []int{1, 2, 5, 10, 20, 50, 100} {
		unmarshal := Unmarshal

		var fields []reflect.StructField
		for i := range n {
			fields = append(fields, reflect.StructField{
				Name: fmt.Sprintf("Name%d", i),
				Type: T[int](),
				Tag:  `json:",case:ignore"`,
			})
		}
		out := reflect.New(reflect.StructOf(fields)).Interface()

		t.Run(fmt.Sprintf("N%d", n), func(t *testing.T) {
			if err := unmarshal(in, out); err != nil {
				t.Fatalf("Unmarshal error: %v", err)
			}
		})
	}
}
