// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "testing"

var amatchTests = []struct {
	pattern string
	name    string
	ok      bool
}{
	{"a", "a", true},
	{"a", "b", false},
	{"a/**", "a", true},
	{"a/**", "b", false},
	{"a/**", "a/b", true},
	{"a/**", "b/b", false},
	{"a/**", "a/b/c/d/e/f", true},
	{"a/**", "z/a/b/c/d/e/f", false},
	{"**/a", "a", true},
	{"**/a", "b", false},
	{"**/a", "x/a", true},
	{"**/a", "x/a/b", false},
	{"**/a", "x/y/z/a", true},
	{"**/a", "x/y/z/a/b", false},

	{"go/pkg/tool/*/compile", "go/pkg/tool/darwin_amd64/compile", true},
}

func TestAmatch(t *testing.T) {
	for _, tt := range amatchTests {
		ok, err := amatch(tt.pattern, tt.name)
		if ok != tt.ok || err != nil {
			t.Errorf("amatch(%q, %q) = %v, %v, want %v, nil", tt.pattern, tt.name, ok, err, tt.ok)
		}
	}
}
