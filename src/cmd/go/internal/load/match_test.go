// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package load

import (
	"strings"
	"testing"
)

var matchPatternTests = `
	pattern ...
	match foo
	
	pattern net
	match net
	not net/http
	
	pattern net/http
	match net/http
	not net
	
	pattern net...
	match net net/http netchan
	not not/http not/net/http
	
	pattern net/...
	match net net/http
	not not/http not/net/http netchan
`

func TestMatchPattern(t *testing.T) {
	testPatterns(t, "matchPattern", matchPatternTests, func(pattern, name string) bool {
		return matchPattern(pattern)(name)
	})
}

var treeCanMatchPatternTests = `
	pattern ...
	match foo
	
	pattern net
	match net
	not net/http
	
	pattern net/http
	match net net/http
	
	pattern net...
	match net netchan net/http
	not not/http not/net/http

	pattern net/...
	match net net/http
	not not/http netchan
	
	pattern abc.../def
	match abcxyz
	not xyzabc
	
	pattern x/y/z/...
	match x x/y x/y/z x/y/z/w
	
	pattern x/y/z
	match x x/y x/y/z
	not x/y/z/w
	
	pattern x/.../y/z
	match x/a/b/c
	not y/x/a/b/c
`

func TestTreeCanMatchPattern(t *testing.T) {
	testPatterns(t, "treeCanMatchPattern", treeCanMatchPatternTests, func(pattern, name string) bool {
		return treeCanMatchPattern(pattern)(name)
	})
}

var hasPathPrefixTests = []stringPairTest{
	{"abc", "a", false},
	{"a/bc", "a", true},
	{"a", "a", true},
	{"a/bc", "a/", true},
}

func TestHasPathPrefix(t *testing.T) {
	testStringPairs(t, "hasPathPrefix", hasPathPrefixTests, hasPathPrefix)
}

type stringPairTest struct {
	in1 string
	in2 string
	out bool
}

func testStringPairs(t *testing.T, name string, tests []stringPairTest, f func(string, string) bool) {
	for _, tt := range tests {
		if out := f(tt.in1, tt.in2); out != tt.out {
			t.Errorf("%s(%q, %q) = %v, want %v", name, tt.in1, tt.in2, out, tt.out)
		}
	}
}

func testPatterns(t *testing.T, name, tests string, fn func(string, string) bool) {
	var patterns []string
	for _, line := range strings.Split(tests, "\n") {
		if i := strings.Index(line, "#"); i >= 0 {
			line = line[:i]
		}
		f := strings.Fields(line)
		if len(f) == 0 {
			continue
		}
		switch f[0] {
		default:
			t.Fatalf("unknown directive %q", f[0])
		case "pattern":
			patterns = f[1:]
		case "match", "not":
			want := f[0] == "match"
			for _, pattern := range patterns {
				for _, in := range f[1:] {
					if fn(pattern, in) != want {
						t.Errorf("%s(%q, %q) = %v, want %v", name, pattern, in, !want, want)
					}
				}
			}
		}
	}
}
