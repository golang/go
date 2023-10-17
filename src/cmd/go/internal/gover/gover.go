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
	"cmp"
)

// A version is a parsed Go version: major[.minor[.patch]][kind[pre]]
// The numbers are the original decimal strings to avoid integer overflows
// and since there is very little actual math. (Probably overflow doesn't matter in practice,
// but at the time this code was written, there was an existing test that used
// go1.99999999999, which does not fit in an int on 32-bit platforms.
// The "big decimal" representation avoids the problem entirely.)
type version struct {
	major string // decimal
	minor string // decimal or ""
	patch string // decimal or ""
	kind  string // "", "alpha", "beta", "rc"
	pre   string // decimal or ""
}

// Compare returns -1, 0, or +1 depending on whether
// x < y, x == y, or x > y, interpreted as toolchain versions.
// The versions x and y must not begin with a "go" prefix: just "1.21" not "go1.21".
// Malformed versions compare less than well-formed versions and equal to each other.
// The language version "1.21" compares less than the release candidate and eventual releases "1.21rc1" and "1.21.0".
func Compare(x, y string) int {
	vx := parse(x)
	vy := parse(y)

	if c := cmpInt(vx.major, vy.major); c != 0 {
		return c
	}
	if c := cmpInt(vx.minor, vy.minor); c != 0 {
		return c
	}
	if c := cmpInt(vx.patch, vy.patch); c != 0 {
		return c
	}
	if c := cmp.Compare(vx.kind, vy.kind); c != 0 { // "" < alpha < beta < rc
		return c
	}
	if c := cmpInt(vx.pre, vy.pre); c != 0 {
		return c
	}
	return 0
}

// Max returns the maximum of x and y interpreted as toolchain versions,
// compared using Compare.
// If x and y compare equal, Max returns x.
func Max(x, y string) string {
	if Compare(x, y) < 0 {
		return y
	}
	return x
}

// Toolchain returns the maximum of x and y interpreted as toolchain names,
// compared using Compare(FromToolchain(x), FromToolchain(y)).
// If x and y compare equal, Max returns x.
func ToolchainMax(x, y string) string {
	if Compare(FromToolchain(x), FromToolchain(y)) < 0 {
		return y
	}
	return x
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
	v := parse(x)
	return v != version{} && v.patch == "" && v.kind == "" && v.pre == ""
}

// Lang returns the Go language version. For example, Lang("1.2.3") == "1.2".
func Lang(x string) string {
	v := parse(x)
	if v.minor == "" {
		return v.major
	}
	return v.major + "." + v.minor
}

// IsPrerelease reports whether v denotes a Go prerelease version.
func IsPrerelease(x string) bool {
	return parse(x).kind != ""
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
	v := parse(x)
	if cmpInt(v.minor, "1") <= 0 {
		return v.major
	}
	return v.major + "." + decInt(v.minor)
}

// IsValid reports whether the version x is valid.
func IsValid(x string) bool {
	return parse(x) != version{}
}

// parse parses the Go version string x into a version.
// It returns the zero version if x is malformed.
func parse(x string) version {
	var v version

	// Parse major version.
	var ok bool
	v.major, x, ok = cutInt(x)
	if !ok {
		return version{}
	}
	if x == "" {
		// Interpret "1" as "1.0.0".
		v.minor = "0"
		v.patch = "0"
		return v
	}

	// Parse . before minor version.
	if x[0] != '.' {
		return version{}
	}

	// Parse minor version.
	v.minor, x, ok = cutInt(x[1:])
	if !ok {
		return version{}
	}
	if x == "" {
		// Patch missing is same as "0" for older versions.
		// Starting in Go 1.21, patch missing is different from explicit .0.
		if cmpInt(v.minor, "21") < 0 {
			v.patch = "0"
		}
		return v
	}

	// Parse patch if present.
	if x[0] == '.' {
		v.patch, x, ok = cutInt(x[1:])
		if !ok || x != "" {
			// Note that we are disallowing prereleases (alpha, beta, rc) for patch releases here (x != "").
			// Allowing them would be a bit confusing because we already have:
			//	1.21 < 1.21rc1
			// But a prerelease of a patch would have the opposite effect:
			//	1.21.3rc1 < 1.21.3
			// We've never needed them before, so let's not start now.
			return version{}
		}
		return v
	}

	// Parse prerelease.
	i := 0
	for i < len(x) && (x[i] < '0' || '9' < x[i]) {
		if x[i] < 'a' || 'z' < x[i] {
			return version{}
		}
		i++
	}
	if i == 0 {
		return version{}
	}
	v.kind, x = x[:i], x[i:]
	if x == "" {
		return v
	}
	v.pre, x, ok = cutInt(x)
	if !ok || x != "" {
		return version{}
	}

	return v
}

// cutInt scans the leading decimal number at the start of x to an integer
// and returns that value and the rest of the string.
func cutInt(x string) (n, rest string, ok bool) {
	i := 0
	for i < len(x) && '0' <= x[i] && x[i] <= '9' {
		i++
	}
	if i == 0 || x[0] == '0' && i != 1 {
		return "", "", false
	}
	return x[:i], x[i:], true
}

// cmpInt returns cmp.Compare(x, y) interpreting x and y as decimal numbers.
// (Copied from golang.org/x/mod/semver's compareInt.)
func cmpInt(x, y string) int {
	if x == y {
		return 0
	}
	if len(x) < len(y) {
		return -1
	}
	if len(x) > len(y) {
		return +1
	}
	if x < y {
		return -1
	} else {
		return +1
	}
}

// decInt returns the decimal string decremented by 1, or the empty string
// if the decimal is all zeroes.
// (Copied from golang.org/x/mod/module's decDecimal.)
func decInt(decimal string) string {
	// Scan right to left turning 0s to 9s until you find a digit to decrement.
	digits := []byte(decimal)
	i := len(digits) - 1
	for ; i >= 0 && digits[i] == '0'; i-- {
		digits[i] = '9'
	}
	if i < 0 {
		// decimal is all zeros
		return ""
	}
	if i == 0 && digits[i] == '1' && len(digits) > 1 {
		digits = digits[1:]
	} else {
		digits[i]--
	}
	return string(digits)
}
