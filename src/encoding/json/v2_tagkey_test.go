// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jsonv2

package json

import "testing"

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
	V string `json:"text/html%"` // https://golang.org/issue/2718
}

type punctuationTag struct {
	V string `json:"!#$%&()*+-./:;<=>?@[]^_{|}~ "` // https://golang.org/issue/3546
}

type dashTag struct {
	V string `json:"-,"`
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

type spaceTag struct {
	Q string `json:"With space"`
}

type unicodeTag struct {
	W string `json:"Ελλάδα"`
}

func TestStructTagObjectKey(t *testing.T) {
	tests := []struct {
		CaseName
		raw   any
		value string
		key   string
	}{
		{Name(""), basicLatin2xTag{"2x"}, "2x", "$%-/"},
		{Name(""), basicLatin3xTag{"3x"}, "3x", "0123456789"},
		{Name(""), basicLatin4xTag{"4x"}, "4x", "ABCDEFGHIJKLMO"},
		{Name(""), basicLatin5xTag{"5x"}, "5x", "PQRSTUVWXYZ_"},
		{Name(""), basicLatin6xTag{"6x"}, "6x", "abcdefghijklmno"},
		{Name(""), basicLatin7xTag{"7x"}, "7x", "pqrstuvwxyz"},
		{Name(""), miscPlaneTag{"いろはにほへと"}, "いろはにほへと", "色は匂へど"},
		{Name(""), dashTag{"foo"}, "foo", "-"},
		{Name(""), emptyTag{"Pour Moi"}, "Pour Moi", "W"},
		{Name(""), misnamedTag{"Animal Kingdom"}, "Animal Kingdom", "X"},
		{Name(""), badFormatTag{"Orfevre"}, "Orfevre", "Y"},
		{Name(""), badCodeTag{"Reliable Man"}, "Reliable Man", "Z"},
		{Name(""), percentSlashTag{"brut"}, "brut", "text/html%"},
		{Name(""), punctuationTag{"Union Rags"}, "Union Rags", "!#$%&()*+-./:;<=>?@[]^_{|}~ "},
		{Name(""), spaceTag{"Perreddu"}, "Perreddu", "With space"},
		{Name(""), unicodeTag{"Loukanikos"}, "Loukanikos", "Ελλάδα"},
	}
	for _, tt := range tests {
		t.Run(tt.Name, func(t *testing.T) {
			b, err := Marshal(tt.raw)
			if err != nil {
				t.Fatalf("%s: Marshal error: %v", tt.Where, err)
			}
			var f any
			err = Unmarshal(b, &f)
			if err != nil {
				t.Fatalf("%s: Unmarshal error: %v", tt.Where, err)
			}
			for k, v := range f.(map[string]any) {
				if k == tt.key {
					if s, ok := v.(string); !ok || s != tt.value {
						t.Fatalf("%s: Unmarshal(%#q) value:\n\tgot:  %q\n\twant: %q", tt.Where, b, s, tt.value)
					}
				} else {
					t.Fatalf("%s: Unmarshal(%#q): unexpected key: %q", tt.Where, b, k)
				}
			}
		})
	}
}
