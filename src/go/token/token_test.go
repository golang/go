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

func TestIsKeyword(t *testing.T) {
	for i := keyword_beg + 1; i < keyword_end; i++ {
		tok, ok := isKeyword(tokens[i])
		if !ok {
			t.Errorf("want true, keyword is %q", tokens[i])
			return
		}
		if tok != i {
			t.Errorf("want equal, have: %d, want: %d", tok, i)
			return
		}
	}
	_, ok := isKeyword("foo")
	if ok {
		t.Errorf("want false, keyword is %q", "foo")
		return
	}
}

func BenchmarkIsKeyword(b *testing.B) {
	keywords := tokens[keyword_beg+1 : keyword_end]
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := 0; j < len(keywords); j++ {
			IsKeyword(keywords[j])
		}
	}
}

func BenchmarkLookup(b *testing.B) {
	keywords := tokens[keyword_beg+1 : keyword_end]
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := 0; j < len(keywords); j++ {
			Lookup(keywords[j])
		}
	}
}

func BenchmarkIsKeyword_WithNonKeywords(b *testing.B) {
	keywords := make([]string, 0, keyword_end-(keyword_beg+1))
	for i := keyword_beg + 1; i < keyword_end; i++ {
		keywords = append(keywords, tokens[i]+"z")
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := 0; j < len(keywords); j++ {
			IsKeyword(keywords[j])
		}
	}
}

func BenchmarkLookup_WithNonKeywords(b *testing.B) {
	keywords := make([]string, 0, keyword_end-(keyword_beg+1))
	for i := keyword_beg + 1; i < keyword_end; i++ {
		keywords = append(keywords, tokens[i]+"z")
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := 0; j < len(keywords); j++ {
			Lookup(keywords[j])
		}
	}
}
