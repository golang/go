// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "testing"

var matchPatternTests = []stringPairTest{
	{"...", "foo", true},
	{"net", "net", true},
	{"net", "net/http", false},
	{"net/http", "net", false},
	{"net/http", "net/http", true},
	{"net...", "netchan", true},
	{"net...", "net", true},
	{"net...", "net/http", true},
	{"net...", "not/http", false},
	{"net/...", "netchan", false},
	{"net/...", "net", true},
	{"net/...", "net/http", true},
	{"net/...", "not/http", false},
}

func TestMatchPattern(t *testing.T) {
	testStringPairs(t, "matchPattern", matchPatternTests, func(pattern, name string) bool {
		return matchPattern(pattern)(name)
	})
}

var treeCanMatchPatternTests = []stringPairTest{
	{"...", "foo", true},
	{"net", "net", true},
	{"net", "net/http", false},
	{"net/http", "net", true},
	{"net/http", "net/http", true},
	{"net...", "netchan", true},
	{"net...", "net", true},
	{"net...", "net/http", true},
	{"net...", "not/http", false},
	{"net/...", "netchan", false},
	{"net/...", "net", true},
	{"net/...", "net/http", true},
	{"net/...", "not/http", false},
	{"abc.../def", "abcxyz", true},
	{"abc.../def", "xyxabc", false},
	{"x/y/z/...", "x", true},
	{"x/y/z/...", "x/y", true},
	{"x/y/z/...", "x/y/z", true},
	{"x/y/z/...", "x/y/z/w", true},
	{"x/y/z", "x", true},
	{"x/y/z", "x/y", true},
	{"x/y/z", "x/y/z", true},
	{"x/y/z", "x/y/z/w", false},
	{"x/.../y/z", "x/a/b/c", true},
	{"x/.../y/z", "y/x/a/b/c", false},
}

func TestChildrenCanMatchPattern(t *testing.T) {
	testStringPairs(t, "treeCanMatchPattern", treeCanMatchPatternTests, func(pattern, name string) bool {
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
