// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os_test

import (
	. "os"
	"slices"
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
	{"start$+middle$^end$", "start$+middle$^end$"},
	{"mixed$|bag$$$", "mixed$|bagPID$"},
	{"$", "$"},
	{"$}", "$}"},
	{"${", ""},  // invalid syntax; eat up the characters
	{"${}", ""}, // invalid syntax; eat up the characters
}

func TestExpand(t *testing.T) {
	for _, test := range expandTests {
		result := Expand(test.in, testGetenv)
		if result != test.out {
			t.Errorf("Expand(%q)=%q; expected %q", test.in, result, test.out)
		}
	}
}

var global any

func BenchmarkExpand(b *testing.B) {
	b.Run("noop", func(b *testing.B) {
		var s string
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			s = Expand("tick tick tick tick", func(string) string { return "" })
		}
		global = s
	})
	b.Run("multiple", func(b *testing.B) {
		var s string
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			s = Expand("$a $a $a $a", func(string) string { return "boom" })
		}
		global = s
	})
}

func TestConsistentEnviron(t *testing.T) {
	e0 := Environ()
	for i := 0; i < 10; i++ {
		e1 := Environ()
		if !slices.Equal(e0, e1) {
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

func TestClearenv(t *testing.T) {
	const testKey = "GO_TEST_CLEARENV"
	const testValue = "1"

	// reset env
	defer func(origEnv []string) {
		for _, pair := range origEnv {
			// Environment variables on Windows can begin with =
			// https://devblogs.microsoft.com/oldnewthing/20100506-00/?p=14133
			i := strings.Index(pair[1:], "=") + 1
			if err := Setenv(pair[:i], pair[i+1:]); err != nil {
				t.Errorf("Setenv(%q, %q) failed during reset: %v", pair[:i], pair[i+1:], err)
			}
		}
	}(Environ())

	if err := Setenv(testKey, testValue); err != nil {
		t.Fatalf("Setenv(%q, %q) failed: %v", testKey, testValue, err)
	}
	if _, ok := LookupEnv(testKey); !ok {
		t.Errorf("Setenv(%q, %q) didn't set $%s", testKey, testValue, testKey)
	}
	Clearenv()
	if val, ok := LookupEnv(testKey); ok {
		t.Errorf("Clearenv() didn't clear $%s, remained with value %q", testKey, val)
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
	_, ok = LookupEnv(smallpox)
	if !ok {
		t.Errorf("smallpox release failed; world remains safe but LookupEnv is broken")
	}
}

// On Windows, Environ was observed to report keys with a single leading "=".
// Check that they are properly reported by LookupEnv and can be set by SetEnv.
// See https://golang.org/issue/49886.
func TestEnvironConsistency(t *testing.T) {
	t.Parallel()

	for _, kv := range Environ() {
		i := strings.Index(kv, "=")
		if i == 0 {
			// We observe in practice keys with a single leading "=" on Windows.
			// TODO(#49886): Should we consume only the first leading "=" as part
			// of the key, or parse through arbitrarily many of them until a non-=,
			// or try each possible key/value boundary until LookupEnv succeeds?
			i = strings.Index(kv[1:], "=") + 1
		}
		if i < 0 {
			t.Errorf("Environ entry missing '=': %q", kv)
		}

		k := kv[:i]
		v := kv[i+1:]
		v2, ok := LookupEnv(k)
		if ok && v == v2 {
			t.Logf("LookupEnv(%q) = %q, %t", k, v2, ok)
		} else {
			t.Errorf("Environ contains %q, but LookupEnv(%q) = %q, %t", kv, k, v2, ok)
		}

		// Since k=v is already present in the environment,
		// setting it should be a no-op.
		if err := Setenv(k, v); err == nil {
			t.Logf("Setenv(%q, %q)", k, v)
		} else {
			t.Errorf("Environ contains %q, but SetEnv(%q, %q) = %q", kv, k, v, err)
		}
	}
}
