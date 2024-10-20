// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"testing"
)

var buildParserTests = []struct {
	x       string
	matched bool
	err     error
}{
	{"gc", true, nil},
	{"gccgo", false, nil},
	{"!gc", false, nil},
	{"gc && gccgo", false, nil},
	{"gc || gccgo", true, nil},
	{"gc || (gccgo && !gccgo)", true, nil},
	{"gc && (gccgo || !gccgo)", true, nil},
	{"!(gc && (gccgo || !gccgo))", false, nil},
	{"gccgo || gc", true, nil},
	{"!(!(!(gccgo || gc)))", false, nil},
	{"compiler_bootstrap", false, nil},
	{"cmd_go_bootstrap", true, nil},
	{"syntax(error", false, fmt.Errorf("parsing //go:build line: unexpected (")},
	{"(gc", false, fmt.Errorf("parsing //go:build line: missing )")},
	{"gc gc", false, fmt.Errorf("parsing //go:build line: unexpected tag")},
	{"(gc))", false, fmt.Errorf("parsing //go:build line: unexpected )")},
}

func TestBuildParser(t *testing.T) {
	for _, tt := range buildParserTests {
		matched, err := matchexpr(tt.x)
		if matched != tt.matched || !equalError(err, tt.err) {
			t.Errorf("matchexpr(%q) = %v, %v; want %v, %v", tt.x, matched, err, tt.matched, tt.err)
		}
	}
}

func equalError(a, b error) bool {
	if a == nil {
		return b == nil
	}
	if b == nil {
		return a == nil
	}
	return a.Error() == b.Error()
}
