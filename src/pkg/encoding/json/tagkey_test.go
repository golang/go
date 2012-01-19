// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package json

import (
	"testing"
)

type basicLatin2xTag struct {
	V string `json:"$%-/"`
}

type basicLatin3xTag struct {
	V string `json:"0123456789"`
}

type basicLatin4xTag struct {
	V string `json:"ABCDEFGHIJKLMO"`
}

type basicLatin5xTag struct {
	V string `json:"PQRSTUVWXYZ_"`
}

type basicLatin6xTag struct {
	V string `json:"abcdefghijklmno"`
}

type basicLatin7xTag struct {
	V string `json:"pqrstuvwxyz"`
}

type miscPlaneTag struct {
	V string `json:"色は匂へど"`
}

type percentSlashTag struct {
	V string `json:"text/html%"` // http://golang.org/issue/2718
}

type emptyTag struct {
	W string
}

type misnamedTag struct {
	X string `jsom:"Misnamed"`
}

type badFormatTag struct {
	Y string `:"BadFormat"`
}

type badCodeTag struct {
	Z string `json:" !\"#&'()*+,."`
}

var structTagObjectKeyTests = []struct {
	raw   interface{}
	value string
	key   string
}{
	{basicLatin2xTag{"2x"}, "2x", "$%-/"},
	{basicLatin3xTag{"3x"}, "3x", "0123456789"},
	{basicLatin4xTag{"4x"}, "4x", "ABCDEFGHIJKLMO"},
	{basicLatin5xTag{"5x"}, "5x", "PQRSTUVWXYZ_"},
	{basicLatin6xTag{"6x"}, "6x", "abcdefghijklmno"},
	{basicLatin7xTag{"7x"}, "7x", "pqrstuvwxyz"},
	{miscPlaneTag{"いろはにほへと"}, "いろはにほへと", "色は匂へど"},
	{emptyTag{"Pour Moi"}, "Pour Moi", "W"},
	{misnamedTag{"Animal Kingdom"}, "Animal Kingdom", "X"},
	{badFormatTag{"Orfevre"}, "Orfevre", "Y"},
	{badCodeTag{"Reliable Man"}, "Reliable Man", "Z"},
	{percentSlashTag{"brut"}, "brut", "text/html%"},
}

func TestStructTagObjectKey(t *testing.T) {
	for _, tt := range structTagObjectKeyTests {
		b, err := Marshal(tt.raw)
		if err != nil {
			t.Fatalf("Marshal(%#q) failed: %v", tt.raw, err)
		}
		var f interface{}
		err = Unmarshal(b, &f)
		if err != nil {
			t.Fatalf("Unmarshal(%#q) failed: %v", b, err)
		}
		for i, v := range f.(map[string]interface{}) {
			switch i {
			case tt.key:
				if s, ok := v.(string); !ok || s != tt.value {
					t.Fatalf("Unexpected value: %#q, want %v", s, tt.value)
				}
			default:
				t.Fatalf("Unexpected key: %#q, from %#q", i, b)
			}
		}
	}
}
