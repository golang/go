// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package base

import (
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"

	"cmd/go/internal/str"
)

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

var cwdOnce = sync.OnceValue(UncachedCwd)

// Cwd returns the current working directory at the time of the first call.
func Cwd() string {
	return cwdOnce()
}

// ShortPath returns an absolute or relative name for path, whatever is shorter.
// There are rare cases where the path produced by ShortPath could be incorrect
// so it should only be used when formatting paths for error messages, not to read
// a file.
func ShortPath(path string) string {
	if rel, err := filepath.Rel(Cwd(), path); err == nil && len(rel) < len(path) {
		return rel
	}
	return path
}

// ShortPathConservative is similar to ShortPath, but returns the input if the result of ShortPath
// would meet conditions that could make it invalid. If the short path would reach into a
// parent directory and the base path contains a symlink, a ".." component can
// cross a symlink boundary. That could be a problem because the symlinks could be evaluated,
// changing the relative location of the boundary, before the ".." terms are applied to
// go to parents. The check here is a little more conservative: it checks
// whether the path starts with a ../ or ..\ component, and if any of the parent directories
// of the working directory are symlinks.
// See #68383 for a case where this could happen.
func ShortPathConservative(path string) string {
	if rel, err := relConservative(Cwd(), path); err == nil && len(rel) < len(path) {
		return rel
	}
	return path
}

func relConservative(basepath, targpath string) (string, error) {
	relpath, err := filepath.Rel(basepath, targpath)
	if err != nil {
		return "", err
	}
	if strings.HasPrefix(relpath, str.WithFilePathSeparator("..")) {
		expanded, err := filepath.EvalSymlinks(basepath)
		if err != nil || expanded != basepath { // The basepath contains a symlink. Be conservative and reject it.
			return "", errors.New("conservatively rejecting relative path that may be invalid")
		}
	}
	return relpath, nil
}

// RelPaths returns a copy of paths with absolute paths
// made relative to the current directory if they would be shorter.
func RelPaths(paths []string) []string {
	var out []string
	for _, p := range paths {
		rel, err := relConservative(Cwd(), p)
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
