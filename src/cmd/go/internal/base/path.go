// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package base

import (
	"os"
	"path/filepath"
	"strings"
	"sync"
)

var cwd string
var cwdOnce sync.Once

// Cwd returns the current working directory at the time of the first call.
func Cwd() string {
	cwdOnce.Do(func() {
		var err error
		cwd, err = os.Getwd()
		if err != nil {
			Fatalf("cannot determine current directory: %v", err)
		}
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
