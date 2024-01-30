// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/version"
	"internal/goversion"
	"strings"
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

// langCompat reports an error if the representation of a numeric
// literal is not compatible with the current language version.
func (check *Checker) langCompat(lit *ast.BasicLit) {
	s := lit.Value
	if len(s) <= 2 || check.allowVersion(check.pkg, lit, go1_13) {
		return
	}
	// len(s) > 2
	if strings.Contains(s, "_") {
		check.versionErrorf(lit, go1_13, "underscores in numeric literals")
		return
	}
	if s[0] != '0' {
		return
	}
	radix := s[1]
	if radix == 'b' || radix == 'B' {
		check.versionErrorf(lit, go1_13, "binary literals")
		return
	}
	if radix == 'o' || radix == 'O' {
		check.versionErrorf(lit, go1_13, "0o/0O-style octal literals")
		return
	}
	if lit.Kind != token.INT && (radix == 'x' || radix == 'X') {
		check.versionErrorf(lit, go1_13, "hexadecimal floating-point literals")
	}
}

// allowVersion reports whether the given package is allowed to use version v.
func (check *Checker) allowVersion(pkg *Package, at positioner, v goVersion) bool {
	// We assume that imported packages have all been checked,
	// so we only have to check for the local package.
	if pkg != check.pkg {
		return true
	}

	// If no explicit file version is specified,
	// fileVersion corresponds to the module version.
	var fileVersion goVersion
	if pos := at.Pos(); pos.IsValid() {
		// We need version.Lang below because file versions
		// can be (unaltered) Config.GoVersion strings that
		// may contain dot-release information.
		fileVersion = asGoVersion(check.versions[check.fileFor(pos)])
	}
	return !fileVersion.isValid() || fileVersion.cmp(v) >= 0
}

// verifyVersionf is like allowVersion but also accepts a format string and arguments
// which are used to report a version error if allowVersion returns false. It uses the
// current package.
func (check *Checker) verifyVersionf(at positioner, v goVersion, format string, args ...interface{}) bool {
	if !check.allowVersion(check.pkg, at, v) {
		check.versionErrorf(at, v, format, args...)
		return false
	}
	return true
}

// TODO(gri) Consider a more direct (position-independent) mechanism
//           to identify which file we're in so that version checks
//           work correctly in the absence of correct position info.

// fileFor returns the *ast.File which contains the position pos.
// If there are no files, the result is nil.
// The position must be valid.
func (check *Checker) fileFor(pos token.Pos) *ast.File {
	assert(pos.IsValid())
	// Eval and CheckExpr tests may not have any source files.
	if len(check.files) == 0 {
		return nil
	}
	for _, file := range check.files {
		if file.FileStart <= pos && pos < file.FileEnd {
			return file
		}
	}
	panic(check.sprintf("file not found for pos = %d (%s)", int(pos), check.fset.Position(pos)))
}
