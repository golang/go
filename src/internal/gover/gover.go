// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package gover implements support for Go toolchain versions like 1.21.0 and 1.21rc1.
// (For historical reasons, Go does not use semver for its toolchains.)
// This package provides the same basic analysis that golang.org/x/mod/semver does for semver.
//
// The go/version package should be imported instead of this one when possible.
// Note that this package works on "1.21" while go/version works on "go1.21".
package gover

import (
	"cmp"
)

// A Version is a parsed Go version: major[.Minor[.Patch]][kind[pre]]
// The numbers are the original decimal strings to avoid integer overflows
// and since there is very little actual math. (Probably overflow doesn't matter in practice,
// but at the time this code was written, there was an existing test that used
// go1.99999999999, which does not fit in an int on 32-bit platforms.
// The "big decimal" representation avoids the problem entirely.)
type Version struct {
	Major string // decimal
	Minor string // decimal or ""
	Patch string // decimal or ""
	Kind  string // "", "alpha", "beta", "rc"
	Pre   string // decimal or ""
}

// Compare returns -1, 0, or +1 depending on whether
// x < y, x == y, or x > y, interpreted as toolchain versions.
// The versions x and y must not begin with a "go" prefix: just "1.21" not "go1.21".
// Malformed versions compare less than well-formed versions and equal to each other.
// The language version "1.21" compares less than the release candidate and eventual releases "1.21rc1" and "1.21.0".
func Compare(x, y string) int {
	vx := Parse(x)
	vy := Parse(y)

	if c := CmpInt(vx.Major, vy.Major); c != 0 {
		return c
	}
	if c := CmpInt(vx.Minor, vy.Minor); c != 0 {
		return c
	}
	if c := CmpInt(vx.Patch, vy.Patch); c != 0 {
		return c
	}
	if c := cmp.Compare(vx.Kind, vy.Kind); c != 0 { // "" < alpha < beta < rc
		// for patch release, alpha < beta < rc < ""
		if vx.Patch != "" {
			if vx.Kind == "" {
				c = 1
			} else if vy.Kind == "" {
				c = -1
			}
		}
		return c
	}
	if c := CmpInt(vx.Pre, vy.Pre); c != 0 {
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
	v := Parse(x)
	return v != Version{} && v.Patch == "" && v.Kind == "" && v.Pre == ""
}

// Lang returns the Go language version. For example, Lang("1.2.3") == "1.2".
func Lang(x string) string {
	v := Parse(x)
	if v.Minor == "" || v.Major == "1" && v.Minor == "0" {
		return v.Major
	}
	return v.Major + "." + v.Minor
}

// IsValid reports whether the version x is valid.
func IsValid(x string) bool {
	return Parse(x) != Version{}
}

// Parse parses the Go version string x into a version.
// It returns the zero version if x is malformed.
func Parse(x string) Version {
	var v Version

	// Parse major version.
	var ok bool
	v.Major, x, ok = cutInt(x)
	if !ok {
		return Version{}
	}
	if x == "" {
		// Interpret "1" as "1.0.0".
		v.Minor = "0"
		v.Patch = "0"
		return v
	}

	// Parse . before minor version.
	if x[0] != '.' {
		return Version{}
	}

	// Parse minor version.
	v.Minor, x, ok = cutInt(x[1:])
	if !ok {
		return Version{}
	}
	if x == "" {
		// Patch missing is same as "0" for older versions.
		// Starting in Go 1.21, patch missing is different from explicit .0.
		if CmpInt(v.Minor, "21") < 0 {
			v.Patch = "0"
		}
		return v
	}

	// Parse patch if present.
	if x[0] == '.' {
		v.Patch, x, ok = cutInt(x[1:])
		if !ok {
			return Version{}
		}

		// If there has prerelease for patch releases.
		if x != "" {
			v.Kind, v.Pre, ok = parsePreRelease(x)
			if !ok {
				return Version{}
			}
		}

		return v
	}

	// Parse prerelease.
	v.Kind, v.Pre, ok = parsePreRelease(x)
	if !ok {
		return Version{}
	}
	return v
}

func parsePreRelease(x string) (kind, pre string, ok bool) {
	i := 0
	for i < len(x) && (x[i] < '0' || '9' < x[i]) {
		if x[i] < 'a' || 'z' < x[i] {
			return "", "", false
		}
		i++
	}
	if i == 0 {
		return "", "", false
	}
	kind, x = x[:i], x[i:]
	if x == "" {
		return kind, "", true
	}
	pre, x, ok = cutInt(x)
	if !ok || x != "" {
		return "", "", false
	}
	return kind, pre, true
}

// cutInt scans the leading decimal number at the start of x to an integer
// and returns that value and the rest of the string.
func cutInt(x string) (n, rest string, ok bool) {
	i := 0
	for i < len(x) && '0' <= x[i] && x[i] <= '9' {
		i++
	}
	if i == 0 || x[0] == '0' && i != 1 { // no digits or unnecessary leading zero
		return "", "", false
	}
	return x[:i], x[i:], true
}

// CmpInt returns cmp.Compare(x, y) interpreting x and y as decimal numbers.
// (Copied from golang.org/x/mod/semver's compareInt.)
func CmpInt(x, y string) int {
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

// DecInt returns the decimal string decremented by 1, or the empty string
// if the decimal is all zeroes.
// (Copied from golang.org/x/mod/module's decDecimal.)
func DecInt(decimal string) string {
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
