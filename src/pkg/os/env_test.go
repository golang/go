// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os_test

import (
	. "os"
	"testing"
)

// testGetenv gives us a controlled set of variables for testing Expand.
func testGetenv(s string) string {
	switch s {
	case "*":
		return "all the args"
	case "#":
		return "NARGS"
	case "$":
		return "PID"
	case "1":
		return "ARGUMENT1"
	case "HOME":
		return "/usr/gopher"
	case "H":
		return "(Value of H)"
	case "home_1":
		return "/usr/foo"
	case "_":
		return "underscore"
	}
	return ""
}

var expandTests = []struct {
	in, out string
}{
	{"", ""},
	{"$*", "all the args"},
	{"$$", "PID"},
	{"${*}", "all the args"},
	{"$1", "ARGUMENT1"},
	{"${1}", "ARGUMENT1"},
	{"now is the time", "now is the time"},
	{"$HOME", "/usr/gopher"},
	{"$home_1", "/usr/foo"},
	{"${HOME}", "/usr/gopher"},
	{"${H}OME", "(Value of H)OME"},
	{"A$$$#$1$H$home_1*B", "APIDNARGSARGUMENT1(Value of H)/usr/foo*B"},
}

func TestExpand(t *testing.T) {
	for _, test := range expandTests {
		result := Expand(test.in, testGetenv)
		if result != test.out {
			t.Errorf("Expand(%q)=%q; expected %q", test.in, result, test.out)
		}
	}
}
