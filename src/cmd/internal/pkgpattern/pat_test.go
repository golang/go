// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pkgpattern

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

	# Special cases. Quoting docs:

	# First, /... at the end of the pattern can match an empty string,
	# so that net/... matches both net and packages in its subdirectories, like net/http.
	pattern net/...
	match net net/http
	not not/http not/net/http netchan

	# Second, any slash-separated pattern element containing a wildcard never
	# participates in a match of the "vendor" element in the path of a vendored
	# package, so that ./... does not match packages in subdirectories of
	# ./vendor or ./mycode/vendor, but ./vendor/... and ./mycode/vendor/... do.
	# Note, however, that a directory named vendor that itself contains code
	# is not a vendored package: cmd/vendor would be a command named vendor,
	# and the pattern cmd/... matches it.
	pattern ./...
	match ./vendor ./mycode/vendor
	not ./vendor/foo ./mycode/vendor/foo

	pattern ./vendor/...
	match ./vendor/foo ./vendor/foo/vendor
	not ./vendor/foo/vendor/bar

	pattern mycode/vendor/...
	match mycode/vendor mycode/vendor/foo mycode/vendor/foo/vendor
	not mycode/vendor/foo/vendor/bar

	pattern x/vendor/y
	match x/vendor/y
	not x/vendor

	pattern x/vendor/y/...
	match x/vendor/y x/vendor/y/z x/vendor/y/vendor x/vendor/y/z/vendor
	not x/vendor/y/vendor/z

	pattern .../vendor/...
	match x/vendor/y x/vendor/y/z x/vendor/y/vendor x/vendor/y/z/vendor
`

func TestMatchPattern(t *testing.T) {
	testPatterns(t, "MatchPattern", matchPatternTests, func(pattern, name string) bool {
		return MatchPattern(pattern)(name)
	})
}

var matchSimplePatternTests = `
	pattern ...
	match foo

	pattern .../bar/.../baz
	match foo/bar/abc/baz

	pattern net
	match net
	not net/http

	pattern net/http
	match net/http
	not net

	pattern net...
	match net net/http netchan
	not not/http not/net/http

	# Special cases. Quoting docs:

	# First, /... at the end of the pattern can match an empty string,
	# so that net/... matches both net and packages in its subdirectories, like net/http.
	pattern net/...
	match net net/http
	not not/http not/net/http netchan
`

func TestSimpleMatchPattern(t *testing.T) {
	testPatterns(t, "MatchSimplePattern", matchSimplePatternTests, func(pattern, name string) bool {
		return MatchSimplePattern(pattern)(name)
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
	testPatterns(t, "TreeCanMatchPattern", treeCanMatchPatternTests, func(pattern, name string) bool {
		return TreeCanMatchPattern(pattern)(name)
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
