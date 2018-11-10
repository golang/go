// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package base

import (
	"os"
	"path/filepath"
	"strings"
)

func getwd() string {
	wd, err := os.Getwd()
	if err != nil {
		Fatalf("cannot determine current directory: %v", err)
	}
	return wd
}

var Cwd = getwd()

// ShortPath returns an absolute or relative name for path, whatever is shorter.
func ShortPath(path string) string {
	if rel, err := filepath.Rel(Cwd, path); err == nil && len(rel) < len(path) {
		return rel
	}
	return path
}

// RelPaths returns a copy of paths with absolute paths
// made relative to the current directory if they would be shorter.
func RelPaths(paths []string) []string {
	var out []string
	// TODO(rsc): Can this use Cwd from above?
	pwd, _ := os.Getwd()
	for _, p := range paths {
		rel, err := filepath.Rel(pwd, p)
		if err == nil && len(rel) < len(p) {
			p = rel
		}
		out = append(out, p)
	}
	return out
}

// FilterDotUnderscoreFiles returns a slice containing all elements
// of path whose base name doesn't begin with "." or "_".
func FilterDotUnderscoreFiles(path []string) []string {
	var out []string // lazily initialized
	for i, p := range path {
		base := filepath.Base(p)
		if strings.HasPrefix(base, ".") || strings.HasPrefix(base, "_") {
			if out == nil {
				out = append(make([]string, 0, len(path)), path[:i]...)
			}
			continue
		}
		if out != nil {
			out = append(out, p)
		}
	}
	if out == nil {
		return path
	}
	return out
}

// IsTestFile reports whether the source file is a set of tests and should therefore
// be excluded from coverage analysis.
func IsTestFile(file string) bool {
	// We don't cover tests, only the code they test.
	return strings.HasSuffix(file, "_test.go")
}
