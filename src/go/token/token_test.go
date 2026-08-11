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
		t.Run(test.name, func(t *testing.T) {
			if got := IsIdentifier(test.in); got != test.want {
				t.Fatalf("IsIdentifier(%q) = %t, want %v", test.in, got, test.want)
			}
		})
	}
}

func TestEnumIsContextualKeyword(t *testing.T) {
	if ENUM.IsKeyword() {
		t.Error("ENUM.IsKeyword() = true, want false")
	}
	if got := Lookup("enum"); got != IDENT {
		t.Errorf("Lookup(\"enum\") = %v, want IDENT", got)
	}
	if IsKeyword("enum") {
		t.Error("IsKeyword(\"enum\") = true, want false")
	}
	if !IsIdentifier("enum") {
		t.Error("IsIdentifier(\"enum\") = false, want true")
	}
	if got := ENUM.String(); got != "enum" {
		t.Errorf("ENUM.String() = %q, want %q", got, "enum")
	}
}
