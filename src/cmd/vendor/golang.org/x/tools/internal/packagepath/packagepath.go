// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package packagepath provides metadata operations on package path
// strings.
package packagepath

// (This package should not depend on go/ast.)
import "strings"

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

// IsStdPackage reports whether the specified package path belongs to a
// package in the standard library (including internal dependencies).
func IsStdPackage(path string) bool {
	// A standard package has no dot in its first segment.
	// (It may yet have a dot, e.g. "vendor/golang.org/x/foo".)
	slash := strings.IndexByte(path, '/')
	if slash < 0 {
		slash = len(path)
	}
	return !strings.Contains(path[:slash], ".") && path != "testdata"
}
