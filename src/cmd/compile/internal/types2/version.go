// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"fmt"
	"go/version"
	"internal/goversion"
)

// A goVersion is a Go language version string of the form "go1.%d"
// where d is the minor version number. goVersion strings don't
// contain release numbers ("go1.20.1" is not a valid goVersion).
type goVersion string

// asGoVersion returns v as a goVersion (e.g., "go1.20.1" becomes "go1.20").
// If v is not a valid Go version, the result is the empty string.
func asGoVersion(v string) goVersion {
	return goVersion(version.Lang(v))
}

// isValid reports whether v is a valid Go version.
func (v goVersion) isValid() bool {
	return v != ""
}

// cmp returns -1, 0, or +1 depending on whether x < y, x == y, or x > y,
// interpreted as Go versions.
func (x goVersion) cmp(y goVersion) int {
	return version.Compare(string(x), string(y))
}

var (
	// Go versions that introduced language changes
	go1_9  = asGoVersion("go1.9")
	go1_13 = asGoVersion("go1.13")
	go1_14 = asGoVersion("go1.14")
	go1_17 = asGoVersion("go1.17")
	go1_18 = asGoVersion("go1.18")
	go1_20 = asGoVersion("go1.20")
	go1_21 = asGoVersion("go1.21")
	go1_22 = asGoVersion("go1.22")
	go1_23 = asGoVersion("go1.23")

	// current (deployed) Go version
	go_current = asGoVersion(fmt.Sprintf("go1.%d", goversion.Version))
)

// allowVersion reports whether the current package at the given position
// is allowed to use version v. If the position is unknown, the specified
// module version (Config.GoVersion) is used. If that version is invalid,
// allowVersion returns true.
func (check *Checker) allowVersion(at poser, v goVersion) bool {
	fileVersion := check.conf.GoVersion
	if pos := at.Pos(); pos.IsKnown() {
		fileVersion = check.versions[pos.FileBase()]
	}

	// We need asGoVersion (which calls version.Lang) below
	// because fileVersion may be the (unaltered) Config.GoVersion
	// string which may contain dot-release information.
	version := asGoVersion(fileVersion)

	return !version.isValid() || version.cmp(v) >= 0
}

// verifyVersionf is like allowVersion but also accepts a format string and arguments
// which are used to report a version error if allowVersion returns false.
func (check *Checker) verifyVersionf(at poser, v goVersion, format string, args ...interface{}) bool {
	if !check.allowVersion(at, v) {
		check.versionErrorf(at, v, format, args...)
		return false
	}
	return true
}
