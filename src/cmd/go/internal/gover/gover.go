// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package gover implements support for Go toolchain versions like 1.21.0 and 1.21rc1.
// (For historical reasons, Go does not use semver for its toolchains.)
// This package provides the same basic analysis that golang.org/x/mod/semver does for semver.
// It also provides some helpers for extracting versions from go.mod files
// and for dealing with module.Versions that may use Go versions or semver
// depending on the module path.
package gover

import (
	"internal/gover"
)

// Compare returns -1, 0, or +1 depending on whether
// x < y, x == y, or x > y, interpreted as toolchain versions.
// The versions x and y must not begin with a "go" prefix: just "1.21" not "go1.21".
// Malformed versions compare less than well-formed versions and equal to each other.
// The language version "1.21" compares less than the release candidate and eventual releases "1.21rc1" and "1.21.0".
func Compare(x, y string) int {
	return gover.Compare(x, y)
}

// Max returns the maximum of x and y interpreted as toolchain versions,
// compared using Compare.
// If x and y compare equal, Max returns x.
func Max(x, y string) string {
	return gover.Max(x, y)
}

// IsLang reports whether v denotes the overall Go language version
// and not a specific release. Starting with the Go 1.21 release, "1.x" denotes
// the overall language version; the first release is "1.x.0".
// The distinction is important because the relative ordering is
//
//	1.21 < 1.21rc1 < 1.21.0
//
// meaning that Go 1.21rc1 and Go 1.21.0 will both handle go.mod files that
// say "go 1.21", but Go 1.21rc1 will not handle files that say "go 1.21.0".
func IsLang(x string) bool {
	return gover.IsLang(x)
}

// Lang returns the Go language version. For example, Lang("1.2.3") == "1.2".
func Lang(x string) string {
	return gover.Lang(x)
}

// IsPrerelease reports whether v denotes a Go prerelease version.
func IsPrerelease(x string) bool {
	return gover.Parse(x).Kind != ""
}

// Prev returns the Go major release immediately preceding v,
// or v itself if v is the first Go major release (1.0) or not a supported
// Go version.
//
// Examples:
//
//	Prev("1.2") = "1.1"
//	Prev("1.3rc4") = "1.2"
func Prev(x string) string {
	v := gover.Parse(x)
	if gover.CmpInt(v.Minor, "1") <= 0 {
		return v.Major
	}
	return v.Major + "." + gover.DecInt(v.Minor)
}

// IsValid reports whether the version x is valid.
func IsValid(x string) bool {
	return gover.IsValid(x)
}
