// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package doc

import (
	"testing"
)

var headingTests = []struct {
	line string
	ok   bool
}{
	{"Section", true},
	{"A typical usage", true},
	{"ΔΛΞ is Greek", true},
	{"Foo 42", true},
	{"", false},
	{"section", false},
	{"A typical usage:", false},
	{"This code:", false},
	{"δ is Greek", false},
	{"Foo §", false},
	{"Fermat's Last Sentence", true},
	{"Fermat's", true},
	{"'sX", false},
	{"Ted 'Too' Bar", false},
	{"Use n+m", false},
	{"Scanning:", false},
	{"N:M", false},
}

func TestIsHeading(t *testing.T) {
	for _, tt := range headingTests {
		if h := heading(tt.line); (len(h) > 0) != tt.ok {
			t.Errorf("isHeading(%q) = %v, want %v", tt.line, h, tt.ok)
		}
	}
}
