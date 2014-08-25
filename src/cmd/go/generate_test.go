// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"reflect"
	"runtime"
	"testing"
)

type splitTest struct {
	in  string
	out []string
}

var splitTests = []splitTest{
	{"", nil},
	{"x", []string{"x"}},
	{" a b\tc ", []string{"a", "b", "c"}},
	{` " a " `, []string{" a "}},
	{"$GOFILE", []string{"proc.go"}},
	{"a $XXNOTDEFINEDXX b", []string{"a", "", "b"}},
	{"/$XXNOTDEFINED/", []string{"//"}},
	{"$GOARCH", []string{runtime.GOARCH}},
	{"yacc -o $GOARCH/yacc_$GOFILE", []string{"go", "tool", "yacc", "-o", runtime.GOARCH + "/yacc_proc.go"}},
}

func TestGenerateCommandParse(t *testing.T) {
	g := &Generator{
		r:        nil, // Unused here.
		path:     "/usr/ken/sys/proc.go",
		dir:      "/usr/ken/sys",
		file:     "proc.go",
		pkg:      "sys",
		commands: make(map[string][]string),
	}
	g.setShorthand([]string{"-command", "yacc", "go", "tool", "yacc"})
	for _, test := range splitTests {
		got := g.split("//go:generate " + test.in)
		if !reflect.DeepEqual(got, test.out) {
			t.Errorf("split(%q): got %q expected %q", test.in, got, test.out)
		}
	}
}
