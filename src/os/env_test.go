// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os_test

import (
	. "os"
	"reflect"
	"strings"
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

func TestConsistentEnviron(t *testing.T) {
	e0 := Environ()
	for i := 0; i < 10; i++ {
		e1 := Environ()
		if !reflect.DeepEqual(e0, e1) {
			t.Fatalf("environment changed")
		}
	}
}

func TestUnsetenv(t *testing.T) {
	const testKey = "GO_TEST_UNSETENV"
	set := func() bool {
		prefix := testKey + "="
		for _, key := range Environ() {
			if strings.HasPrefix(key, prefix) {
				return true
			}
		}
		return false
	}
	if err := Setenv(testKey, "1"); err != nil {
		t.Fatalf("Setenv: %v", err)
	}
	if !set() {
		t.Error("Setenv didn't set TestUnsetenv")
	}
	if err := Unsetenv(testKey); err != nil {
		t.Fatalf("Unsetenv: %v", err)
	}
	if set() {
		t.Fatal("Unsetenv didn't clear TestUnsetenv")
	}
}

func TestLookupEnv(t *testing.T) {
	const smallpox = "SMALLPOX"      // No one has smallpox.
	value, ok := LookupEnv(smallpox) // Should not exist.
	if ok || value != "" {
		t.Fatalf("%s=%q", smallpox, value)
	}
	defer Unsetenv(smallpox)
	err := Setenv(smallpox, "virus")
	if err != nil {
		t.Fatalf("failed to release smallpox virus")
	}
	value, ok = LookupEnv(smallpox)
	if !ok {
		t.Errorf("smallpox release failed; world remains safe but LookupEnv is broken")
	}
}
