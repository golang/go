// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package doc

import (
	"reflect"
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

var blocksTests = []struct {
	in  string
	out []block
}{
	{
		in: `Para 1.
Para 1 line 2.

Para 2.

Section

Para 3.

	pre
	pre1

Para 4.
	pre
	pre2
`,
		out: []block{
			{opPara, []string{"Para 1.\n", "Para 1 line 2.\n"}},
			{opPara, []string{"Para 2.\n"}},
			{opHead, []string{"Section"}},
			{opPara, []string{"Para 3.\n"}},
			{opPre, []string{"pre\n", "pre1\n"}},
			{opPara, []string{"Para 4.\n"}},
			{opPre, []string{"pre\n", "pre2\n"}},
		},
	},
}

func TestBlocks(t *testing.T) {
	for i, tt := range blocksTests {
		b := blocks(tt.in)
		if !reflect.DeepEqual(b, tt.out) {
			t.Errorf("#%d: mismatch\nhave: %v\nwant: %v", i, b, tt.out)
		}
	}
}
