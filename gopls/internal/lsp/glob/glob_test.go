// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package glob_test

import (
	"testing"

	"golang.org/x/tools/gopls/internal/lsp/glob"
)

func TestParseErrors(t *testing.T) {
	tests := []string{
		"***",
		"ab{c",
		"[]",
		"[a-]",
		"ab{c{d}",
	}

	for _, test := range tests {
		_, err := glob.Parse(test)
		if err == nil {
			t.Errorf("Parse(%q) succeeded unexpectedly", test)
		}
	}
}

func TestMatch(t *testing.T) {
	tests := []struct {
		pattern, input string
		want           bool
	}{
		// Basic cases.
		{"", "", true},
		{"", "a", false},
		{"", "/", false},
		{"abc", "abc", true},

		// ** behavior
		{"**", "abc", true},
		{"**/abc", "abc", true},
		{"**", "abc/def", true},
		{"{a/**/c,a/**/d}", "a/b/c", true},
		{"{a/**/c,a/**/d}", "a/b/c/d", true},
		{"{a/**/c,a/**/e}", "a/b/c/d", false},
		{"{a/**/c,a/**/e,a/**/d}", "a/b/c/d", true},
		{"{/a/**/c,a/**/e,a/**/d}", "a/b/c/d", true},
		{"{/a/**/c,a/**/e,a/**/d}", "/a/b/c/d", false},
		{"{/a/**/c,a/**/e,a/**/d}", "/a/b/c", true},
		{"{/a/**/e,a/**/e,a/**/d}", "/a/b/c", false},

		// * and ? behavior
		{"/*", "/a", true},
		{"*", "foo", true},
		{"*o", "foo", true},
		{"*o", "foox", false},
		{"f*o", "foo", true},
		{"f*o", "fo", true},
		{"fo?", "foo", true},
		{"fo?", "fox", true},
		{"fo?", "fooo", false},
		{"fo?", "fo", false},
		{"?", "a", true},
		{"?", "ab", false},
		{"?", "", false},
		{"*?", "", false},
		{"?b", "ab", true},
		{"?c", "ab", false},

		// {} behavior
		{"ab{c,d}e", "abce", true},
		{"ab{c,d}e", "abde", true},
		{"ab{c,d}e", "abxe", false},
		{"ab{c,d}e", "abe", false},
		{"{a,b}c", "ac", true},
		{"{a,b}c", "bc", true},
		{"{a,b}c", "ab", false},
		{"a{b,c}", "ab", true},
		{"a{b,c}", "ac", true},
		{"a{b,c}", "bc", false},
		{"ab{c{1,2},d}e", "abc1e", true},
		{"ab{c{1,2},d}e", "abde", true},
		{"ab{c{1,2},d}e", "abc1f", false},
		{"ab{c{1,2},d}e", "abce", false},
		{"ab{c[}-~]}d", "abc}d", true},
		{"ab{c[}-~]}d", "abc~d", true},
		{"ab{c[}-~],y}d", "abcxd", false},
		{"ab{c[}-~],y}d", "abyd", true},
		{"ab{c[}-~],y}d", "abd", false},
		{"{a/b/c,d/e/f}", "a/b/c", true},
		{"/ab{/c,d}e", "/ab/ce", true},
		{"/ab{/c,d}e", "/ab/cf", false},

		// [-] behavior
		{"[a-c]", "a", true},
		{"[a-c]", "b", true},
		{"[a-c]", "c", true},
		{"[a-c]", "d", false},
		{"[a-c]", " ", false},

		// Realistic examples.
		{"**/*.{ts,js}", "path/to/foo.ts", true},
		{"**/*.{ts,js}", "path/to/foo.js", true},
		{"**/*.{ts,js}", "path/to/foo.go", false},
	}

	for _, test := range tests {
		g, err := glob.Parse(test.pattern)
		if err != nil {
			t.Fatalf("New(%q) failed unexpectedly: %v", test.pattern, err)
		}
		if got := g.Match(test.input); got != test.want {
			t.Errorf("New(%q).Match(%q) = %t, want %t", test.pattern, test.input, got, test.want)
		}
	}
}
