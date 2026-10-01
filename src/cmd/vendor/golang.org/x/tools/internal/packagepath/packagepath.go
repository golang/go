// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package packagepath provides metadata operations on package path
// strings.
package packagepath

// (This package should not depend on go/ast.)
import (
	pathpkg "path"
	"strings"
)

// CanImport reports whether one package is allowed to import another.
//
// TODO(adonovan): allow customization of the accessibility relation
// (e.g. for Bazel).
func CanImport(from, to string) bool {
	// TODO(adonovan): better segment hygiene.
	if to == "internal" || strings.HasPrefix(to, "internal/") {
		// Special case: only std packages may import internal/...
		// We can't reliably know whether we're in std, so we
		// use a heuristic on the first segment.
		first, _, _ := strings.Cut(from, "/")
		if strings.Contains(first, ".") {
			return false // example.com/foo ∉ std
		}
		if first == "testdata" {
			return false // testdata/foo ∉ std
		}
	}
	if strings.HasSuffix(to, "/internal") {
		return strings.HasPrefix(from, to[:len(to)-len("/internal")])
	}
	if i := strings.LastIndex(to, "/internal/"); i >= 0 {
		return strings.HasPrefix(from, to[:i])
	}
	return true
}

// MaybeStdPackage reports whether the specified package path might
// belong to a package in the standard library (including internal
// dependencies), based only on its form.
//
// It may spuriously return true, but a result of false is definitive:
//
//	MaybeStdPackage("fmt")             = true
//	MaybeStdPackage("maybe/tomorrow")  = true  // false positive
//	MaybeStdPackage("example.com/foo") = false
//
// For a definitive answer, use [stdlib.HasPackage], which consults a
// huge table.
func MaybeStdPackage(path string) bool {
	// A standard package has no dot in its first segment.
	// (It may yet have a dot, e.g. "vendor/golang.org/x/foo".)
	slash := strings.IndexByte(path, '/')
	if slash < 0 {
		slash = len(path)
	}
	return !strings.Contains(path[:slash], ".") && path != "testdata"
}

// TrimVersionSuffix removes a possible trailing "/v2" (etc) suffix from a
// package or module path.
//
// This is only a heuristic as to the package's declared name, and
// should only be used for stylistic decisions, such as whether it
// would be clearer to use an explicit local name in the import
// because the declared name differs from the result of this function.
//
// TODO(hxjiang): consider trim ".v3" when using gopkg.in/foo.v2/path/to/package.v3.
func TrimVersionSuffix(path string) string {
	dir, base := pathpkg.Split(path)
	if dir == "" {
		return path
	}

	if len(base) > 1 && base[0] == 'v' && strings.Trim(base[1:], "0123456789") == "" {
		return strings.TrimSuffix(dir, "/") // sans "/v2"
	}
	return path
}
