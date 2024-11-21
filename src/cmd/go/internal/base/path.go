// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package base

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
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
// ShortPath should only be used when formatting paths for error messages.
func ShortPath(path string) string {
	if rel, err := filepath.Rel(Cwd(), path); err == nil && len(rel) < len(path) && sameFile(rel, path) {
		return rel
	}
	return path
}

func sameFile(path1, path2 string) bool {
	fi1, err1 := os.Stat(path1)
	fi2, err2 := os.Stat(path2)
	if err1 != nil || err2 != nil {
		// If there were errors statting the files return false,
		// unless both of the files don't exist.
		return os.IsNotExist(err1) && os.IsNotExist(err2)
	}
	return os.SameFile(fi1, fi2)
}

// ShortPathError rewrites the path in err using base.ShortPath, if err is a wrapped PathError.
func ShortPathError(err error) error {
	var pe *fs.PathError
	if errors.As(err, &pe) {
		pe.Path = ShortPath(pe.Path)
	}
	return err
}

// RelPaths returns a copy of paths with absolute paths
// made relative to the current directory if they would be shorter.
func RelPaths(paths []string) []string {
	out := make([]string, 0, len(paths))
	for _, p := range paths {
		out = append(out, ShortPath(p))
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
