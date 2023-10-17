// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fs_test

import (
	. "io/fs"
	"testing"
)

var isValidPathTests = []struct {
	name string
	ok   bool
}{
	{".", true},
	{"x", true},
	{"x/y", true},

	{"", false},
	{"..", false},
	{"/", false},
	{"x/", false},
	{"/x", false},
	{"x/y/", false},
	{"/x/y", false},
	{"./", false},
	{"./x", false},
	{"x/.", false},
	{"x/./y", false},
	{"../", false},
	{"../x", false},
	{"x/..", false},
	{"x/../y", false},
	{"x//y", false},
	{`x\`, true},
	{`x\y`, true},
	{`x:y`, true},
	{`\x`, true},
}

func TestValidPath(t *testing.T) {
	for _, tt := range isValidPathTests {
		ok := ValidPath(tt.name)
		if ok != tt.ok {
			t.Errorf("ValidPath(%q) = %v, want %v", tt.name, ok, tt.ok)
		}
	}
}
