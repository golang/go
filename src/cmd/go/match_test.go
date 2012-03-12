// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "testing"

var matchTests = []struct {
	pattern string
	path    string
	match   bool
}{
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
	for _, tt := range matchTests {
		match := matchPattern(tt.pattern)(tt.path)
		if match != tt.match {
			t.Errorf("matchPattern(%q)(%q) = %v, want %v", tt.pattern, tt.path, match, tt.match)
		}
	}
}
