// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http2

import (
	"flag"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"
)

var knownFailing = flag.Bool("known_failing", false, "Run known-failing tests.")

func condSkipFailingTest(t *testing.T) {
	if !*knownFailing {
		t.Skip("Skipping known-failing test without --known_failing")
	}
}

func init() {
	DebugGoroutines = true
	flag.BoolVar(&VerboseLogs, "verboseh2", VerboseLogs, "Verbose HTTP/2 debug logging")
}

func TestSettingString(t *testing.T) {
	tests := []struct {
		s    Setting
		want string
	}{
		{Setting{SettingMaxFrameSize, 123}, "[MAX_FRAME_SIZE = 123]"},
		{Setting{1<<16 - 1, 123}, "[UNKNOWN_SETTING_65535 = 123]"},
	}
	for i, tt := range tests {
		got := fmt.Sprint(tt.s)
		if got != tt.want {
			t.Errorf("%d. for %#v, string = %q; want %q", i, tt.s, got, tt.want)
		}
	}
}

func TestSorterPoolAllocs(t *testing.T) {
	ss := []string{"a", "b", "c"}
	h := http.Header{
		"a": nil,
		"b": nil,
		"c": nil,
	}
	sorter := new(sorter)

	if allocs := testing.AllocsPerRun(100, func() {
		sorter.SortStrings(ss)
	}); allocs >= 1 {
		t.Logf("SortStrings allocs = %v; want <1", allocs)
	}

	if allocs := testing.AllocsPerRun(5, func() {
		if len(sorter.Keys(h)) != 3 {
			t.Fatal("wrong result")
		}
	}); allocs > 0 {
		t.Logf("Keys allocs = %v; want <1", allocs)
	}
}

// waitCondition reports whether fn eventually returned true,
// checking immediately and then every checkEvery amount,
// until waitFor has elapsed, at which point it returns false.
func waitCondition(waitFor, checkEvery time.Duration, fn func() bool) bool {
	deadline := time.Now().Add(waitFor)
	for time.Now().Before(deadline) {
		if fn() {
			return true
		}
		time.Sleep(checkEvery)
	}
	return false
}

// waitErrCondition is like waitCondition but with errors instead of bools.
func waitErrCondition(waitFor, checkEvery time.Duration, fn func() error) error {
	deadline := time.Now().Add(waitFor)
	var err error
	for time.Now().Before(deadline) {
		if err = fn(); err == nil {
			return nil
		}
		time.Sleep(checkEvery)
	}
	return err
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

// Tests that http2.Server.IdleTimeout is initialized from
// http.Server.{Idle,Read}Timeout. http.Server.IdleTimeout was
// added in Go 1.8.
func TestConfigureServerIdleTimeout_Go18(t *testing.T) {
	const timeout = 5 * time.Second
	const notThisOne = 1 * time.Second

	// With a zero http2.Server, verify that it copies IdleTimeout:
	{
		s1 := &http.Server{
			IdleTimeout: timeout,
			ReadTimeout: notThisOne,
		}
		s2 := &Server{}
		if err := ConfigureServer(s1, s2); err != nil {
			t.Fatal(err)
		}
		if s2.IdleTimeout != timeout {
			t.Errorf("s2.IdleTimeout = %v; want %v", s2.IdleTimeout, timeout)
		}
	}

	// And that it falls back to ReadTimeout:
	{
		s1 := &http.Server{
			ReadTimeout: timeout,
		}
		s2 := &Server{}
		if err := ConfigureServer(s1, s2); err != nil {
			t.Fatal(err)
		}
		if s2.IdleTimeout != timeout {
			t.Errorf("s2.IdleTimeout = %v; want %v", s2.IdleTimeout, timeout)
		}
	}

	// Verify that s1's IdleTimeout doesn't overwrite an existing setting:
	{
		s1 := &http.Server{
			IdleTimeout: notThisOne,
		}
		s2 := &Server{
			IdleTimeout: timeout,
		}
		if err := ConfigureServer(s1, s2); err != nil {
			t.Fatal(err)
		}
		if s2.IdleTimeout != timeout {
			t.Errorf("s2.IdleTimeout = %v; want %v", s2.IdleTimeout, timeout)
		}
	}
}

var forbiddenStringsFunctions = map[string]bool{
	// Functions that use Unicode-aware case folding.
	"EqualFold":      true,
	"Title":          true,
	"ToLower":        true,
	"ToLowerSpecial": true,
	"ToTitle":        true,
	"ToTitleSpecial": true,
	"ToUpper":        true,
	"ToUpperSpecial": true,

	// Functions that use Unicode-aware spaces.
	"Fields":    true,
	"TrimSpace": true,
}

// TestNoUnicodeStrings checks that nothing in net/http uses the Unicode-aware
// strings and bytes package functions. HTTP is mostly ASCII based, and doing
// Unicode-aware case folding or space stripping can introduce vulnerabilities.
func TestNoUnicodeStrings(t *testing.T) {
	re := regexp.MustCompile(`(strings|bytes).([A-Za-z]+)`)
	if err := filepath.Walk(".", func(path string, info os.FileInfo, err error) error {
		if err != nil {
			t.Fatal(err)
		}

		if path == "h2i" || path == "h2c" {
			return filepath.SkipDir
		}
		if !strings.HasSuffix(path, ".go") ||
			strings.HasSuffix(path, "_test.go") ||
			path == "ascii.go" || info.IsDir() {
			return nil
		}

		contents, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		for lineNum, line := range strings.Split(string(contents), "\n") {
			for _, match := range re.FindAllStringSubmatch(line, -1) {
				if !forbiddenStringsFunctions[match[2]] {
					continue
				}
				t.Errorf("disallowed call to %s at %s:%d", match[0], path, lineNum+1)
			}
		}

		return nil
	}); err != nil {
		t.Fatal(err)
	}
}

// SetForTest sets *p = v, and restores its original value in t.Cleanup.
func SetForTest[T any](t testing.TB, p *T, v T) {
	orig := *p
	t.Cleanup(func() {
		*p = orig
	})
	*p = v
}

// Must returns v if err is nil, or panics otherwise.
func Must[T any](v T, err error) T {
	if err != nil {
		panic(err)
	}
	return v
}
