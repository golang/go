// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This is a fork of internal/gover for use by x/tools until
// go1.21 and earlier are no longer supported by x/tools.

package versions

import "strings"

// A gover is a parsed Go gover: major[.Minor[.Patch]][kind[pre]]
// The numbers are the original decimal strings to avoid integer overflows
// and since there is very little actual math. (Probably overflow doesn't matter in practice,
// but at the time this code was written, there was an existing test that used
// go1.99999999999, which does not fit in an int on 32-bit platforms.
// The "big decimal" representation avoids the problem entirely.)
type gover struct {
	major string // decimal
	minor string // decimal or ""
	patch string // decimal or ""
	kind  string // "", "alpha", "beta", "rc"
	pre   string // decimal or ""
}

// compare returns -1, 0, or +1 depending on whether
// x < y, x == y, or x > y, interpreted as toolchain versions.
// The versions x and y must not begin with a "go" prefix: just "1.21" not "go1.21".
// Malformed versions compare less than well-formed versions and equal to each other.
// The language version "1.21" compares less than the release candidate and eventual releases "1.21rc1" and "1.21.0".
func compare(x, y string) int {
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
	if c := strings.Compare(vx.kind, vy.kind); c != 0 { // "" < alpha < beta < rc
		return c
	}
	if c := cmpInt(vx.pre, vy.pre); c != 0 {
		return c
	}
	return 0
}

// lang returns the Go language version. For example, lang("1.2.3") == "1.2".
func lang(x string) string {
	v := parse(x)
	if v.minor == "" || v.major == "1" && v.minor == "0" {
		return v.major
	}
	return v.major + "." + v.minor
}

// isValid reports whether the version x is valid.
func isValid(x string) bool {
	return parse(x) != gover{}
}

// parse parses the Go version string x into a version.
// It returns the zero version if x is malformed.
func parse(x string) gover {
	var v gover

	// Parse major version.
	var ok bool
	v.major, x, ok = cutInt(x)
	if !ok {
		return gover{}
	}
	if x == "" {
		// Interpret "1" as "1.0.0".
		v.minor = "0"
		v.patch = "0"
		return v
	}

	// Parse . before minor version.
	if x[0] != '.' {
		return gover{}
	}

	// Parse minor version.
	v.minor, x, ok = cutInt(x[1:])
	if !ok {
		return gover{}
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
			return gover{}
		}
		return v
	}

	// Parse prerelease.
	i := 0
	for i < len(x) && (x[i] < '0' || '9' < x[i]) {
		if x[i] < 'a' || 'z' < x[i] {
			return gover{}
		}
		i++
	}
	if i == 0 {
		return gover{}
	}
	v.kind, x = x[:i], x[i:]
	if x == "" {
		return v
	}
	v.pre, x, ok = cutInt(x)
	if !ok || x != "" {
		return gover{}
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
	if i == 0 || x[0] == '0' && i != 1 { // no digits or unnecessary leading zero
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
