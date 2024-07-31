// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package version provides operations on [Go versions]
// in [Go toolchain name syntax]: strings like
// "go1.20", "go1.21.0", "go1.22rc2", and "go1.23.4-bigcorp".
//
// [Go versions]: https://go.dev/doc/toolchain#version
// [Go toolchain name syntax]: https://go.dev/doc/toolchain#name
package version // import "go/version"

import (
	"internal/gover"
	"strings"
)

// stripGo converts from a "go1.21-bigcorp" version to a "1.21" version.
// If v does not start with "go", stripGo returns the empty string (a known invalid version).
func stripGo(v string) string {
	v, _, _ = strings.Cut(v, "-") // strip -bigcorp suffix.
	if len(v) < 2 || v[:2] != "go" {
		return ""
	}
	return v[2:]
}

// Lang returns the Go language version for version x.
// If x is not a valid version, Lang returns the empty string.
// For example:
//
//	Lang("go1.21rc2") = "go1.21"
//	Lang("go1.21.2") = "go1.21"
//	Lang("go1.21") = "go1.21"
//	Lang("go1") = "go1"
//	Lang("bad") = ""
//	Lang("1.21") = ""
func Lang(x string) string {
	v := gover.Lang(stripGo(x))
	if v == "" {
		return ""
	}
	if strings.HasPrefix(x[2:], v) {
		return x[:2+len(v)] // "go"+v without allocation
	} else {
		return "go" + v
	}
}

// Compare returns -1, 0, or +1 depending on whether
// x < y, x == y, or x > y, interpreted as Go versions.
// The versions x and y must begin with a "go" prefix: "go1.21" not "1.21".
// Invalid versions, including the empty string, compare less than
// valid versions and equal to each other.
// After go1.21, the language version is less than specific release versions
// or other prerelease versions.
// For example:
//
//	Compare("go1.21rc1", "go1.21") = 1
//	Compare("go1.21rc1", "go1.21.0") = -1
//	Compare("go1.22rc1", "go1.22") = 1
//	Compare("go1.22rc1", "go1.22.0") = -1
//
// However, When the language version is below go1.21, the situation is quite different,
// because the initial release version was 1.N, not 1.N.0.
// For example:
//
//	Compare("go1.20rc1", "go1.21") = -1
//	Compare("go1.19rc1", "go1.19") = -1
//	Compare("go1.18", "go1.18rc1") = 1
//	Compare("go1.18", "go1.18rc1") = 1
//
// This situation also happens to prerelease for some old patch versions, such as "go1.8.5rc5, "go1.9.2rc2"
// For example:
//
//	Compare("go1.8.5rc4", "go1.8.5rc5") = -1
//	Compare("go1.8.5rc5", "go1.8.5") = -1
//	Compare("go1.9.2rc2", "go1.9.2") = -1
//	Compare("go1.9.2rc2", "go1.9") = 1
func Compare(x, y string) int {
	return gover.Compare(stripGo(x), stripGo(y))
}

// IsValid reports whether the version x is valid.
func IsValid(x string) bool {
	return gover.IsValid(stripGo(x))
}
