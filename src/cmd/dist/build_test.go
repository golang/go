// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"internal/platform"
	"io/fs"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// TestMustLinkExternal verifies that the mustLinkExternal helper
// function matches internal/platform.MustLinkExternal.
func TestMustLinkExternal(t *testing.T) {
	for _, goos := range okgoos {
		for _, goarch := range okgoarch {
			for _, cgoEnabled := range []bool{true, false} {
				got := mustLinkExternal(goos, goarch, cgoEnabled)
				want := platform.MustLinkExternal(goos, goarch, cgoEnabled)
				if got != want {
					t.Errorf("mustLinkExternal(%q, %q, %v) = %v; want %v", goos, goarch, cgoEnabled, got, want)
				}
			}
		}
	}
}

func TestRequiredBootstrapVersion(t *testing.T) {
	testCases := map[string]string{
		"1.22": "1.20",
		"1.23": "1.20",
		"1.24": "1.22",
		"1.25": "1.22",
		"1.26": "1.24",
		"1.27": "1.24",
	}

	for v, want := range testCases {
		if got := requiredBootstrapVersion(v); got != want {
			t.Errorf("requiredBootstrapVersion(%v): got %v, want %v", v, got, want)
		}
	}
}

func TestNoSourceFileNamedLikeATest(t *testing.T) {
	err := filepath.WalkDir(filepath.Join(runtime.GOROOT(), "src", "cmd"), func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		name := d.Name()
		if d.IsDir() && name == "testdata" {
			return filepath.SkipDir
		}
		if strings.HasSuffix(name, ".go") && !strings.HasSuffix(name, "_test.go") && strings.Contains(name, "_test") {
			t.Errorf("%s: shouldbuild omits every file whose name contains _test", path)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}
