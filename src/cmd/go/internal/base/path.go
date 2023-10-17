// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package base

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
)

var cwd string
var cwdOnce sync.Once

// UncachedCwd returns the current working directory.
// Most callers should use Cwd, which caches the result for future use.
// UncachedCwd is appropriate to call early in program startup before flag parsing,
// because the -C flag may change the current directory.
func UncachedCwd() string {
	wd, err := os.Getwd()
	if err != nil {
		Fatalf("cannot determine current directory: %v", err)
	}
	return wd
}

// Cwd returns the current working directory at the time of the first call.
func Cwd() string {
	cwdOnce.Do(func() {
		cwd = UncachedCwd()
	})
	return cwd
}

// ShortPath returns an absolute or relative name for path, whatever is shorter.
func ShortPath(path string) string {
	if rel, err := filepath.Rel(Cwd(), path); err == nil && len(rel) < len(path) {
		return rel
	}
	return path
}

// RelPaths returns a copy of paths with absolute paths
// made relative to the current directory if they would be shorter.
func RelPaths(paths []string) []string {
	var out []string
	for _, p := range paths {
		rel, err := filepath.Rel(Cwd(), p)
		if err == nil && len(rel) < len(p) {
			p = rel
		}
		out = append(out, p)
	}
	return out
}

// IsTestFile reports whether the source file is a set of tests and should therefore
// be excluded from coverage analysis.
func IsTestFile(file string) bool {
	// We don't cover tests, only the code they test.
	return strings.HasSuffix(file, "_test.go")
}

// IsNull reports whether the path is a common name for the null device.
// It returns true for /dev/null on Unix, or NUL (case-insensitive) on Windows.
func IsNull(path string) bool {
	if path == os.DevNull {
		return true
	}
	if runtime.GOOS == "windows" {
		if strings.EqualFold(path, "NUL") {
			return true
		}
	}
	return false
}
