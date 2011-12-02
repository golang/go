// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ast

import (
	"bytes"
	"strings"
	"testing"
)

var tests = []struct {
	x interface{} // x is printed as s
	s string
}{
	// basic types
	{nil, "0  nil"},
	{true, "0  true"},
	{42, "0  42"},
	{3.14, "0  3.14"},
	{1 + 2.718i, "0  (1+2.718i)"},
	{"foobar", "0  \"foobar\""},

	// maps
	{map[string]int{"a": 1},
		`0  map[string]int (len = 1) {
		1  .  "a": 1
		2  }`},

	// pointers
	{new(int), "0  *0"},

	// slices
	{[]int{1, 2, 3},
		`0  []int (len = 3) {
		1  .  0: 1
		2  .  1: 2
		3  .  2: 3
		4  }`},

	// structs
	{struct{ X, Y int }{42, 991},
		`0  struct { X int; Y int } {
		1  .  X: 42
		2  .  Y: 991
		3  }`},
}

// Split s into lines, trim whitespace from all lines, and return
// the concatenated non-empty lines.
func trim(s string) string {
	lines := strings.Split(s, "\n")
	i := 0
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line != "" {
			lines[i] = line
			i++
		}
	}
	return strings.Join(lines[0:i], "\n")
}

func TestPrint(t *testing.T) {
	var buf bytes.Buffer
	for _, test := range tests {
		buf.Reset()
		if _, err := Fprint(&buf, nil, test.x, nil); err != nil {
			t.Errorf("Fprint failed: %s", err)
		}
		if s, ts := trim(buf.String()), trim(test.s); s != ts {
			t.Errorf("got:\n%s\nexpected:\n%s\n", s, ts)
		}
	}
}
