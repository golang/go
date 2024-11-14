// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package token

import "testing"

func TestIsIdentifier(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want bool
	}{
		{"Empty", "", false},
		{"Space", " ", false},
		{"SpaceSuffix", "foo ", false},
		{"Number", "123", false},
		{"Keyword", "func", false},

		{"LettersASCII", "foo", true},
		{"MixedASCII", "_bar123", true},
		{"UppercaseKeyword", "Func", true},
		{"LettersUnicode", "fóö", true},
	}
	for _, test := range tests {
		t.Run(test.name, func { t ->
			if got := IsIdentifier(test.in); got != test.want {
				t.Fatalf("IsIdentifier(%q) = %t, want %v", test.in, got, test.want)
			}
		})
	}
}
