// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.18
// +build go1.18

// Package semver provides shared utilities for manipulating
// Go semantic versions.
package semver

import (
	"regexp"
	"strings"
)

// addSemverPrefix adds a 'v' prefix to s if it isn't already prefixed
// with 'v' or 'go'. This allows us to easily test go-style SEMVER
// strings against normal SEMVER strings.
func addSemverPrefix(s string) string {
	if !strings.HasPrefix(s, "v") && !strings.HasPrefix(s, "go") {
		return "v" + s
	}
	return s
}

// removeSemverPrefix removes the 'v' or 'go' prefixes from go-style
// SEMVER strings, for usage in the public vulnerability format.
func removeSemverPrefix(s string) string {
	s = strings.TrimPrefix(s, "v")
	s = strings.TrimPrefix(s, "go")
	return s
}

// CanonicalizeSemverPrefix turns a SEMVER string into the canonical
// representation using the 'v' prefix, as used by the OSV format.
// Input may be a bare SEMVER ("1.2.3"), Go prefixed SEMVER ("go1.2.3"),
// or already canonical SEMVER ("v1.2.3").
func CanonicalizeSemverPrefix(s string) string {
	return addSemverPrefix(removeSemverPrefix(s))
}

var (
	// Regexp for matching go tags. The groups are:
	// 1  the major.minor version
	// 2  the patch version, or empty if none
	// 3  the entire prerelease, if present
	// 4  the prerelease type ("beta" or "rc")
	// 5  the prerelease number
	tagRegexp = regexp.MustCompile(`^go(\d+\.\d+)(\.\d+|)((beta|rc|-pre)(\d+))?$`)
)
