// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"errors"
	"fmt"
	"go/ast"
	"go/token"
	"strings"
)

// A version represents a released Go version.
type version struct {
	major, minor int
}

func (v version) String() string {
	return fmt.Sprintf("go%d.%d", v.major, v.minor)
}

func (v version) equal(u version) bool {
	return v.major == u.major && v.minor == u.minor
}

func (v version) before(u version) bool {
	return v.major < u.major || v.major == u.major && v.minor < u.minor
}

func (v version) after(u version) bool {
	return v.major > u.major || v.major == u.major && v.minor > u.minor
}

// Go versions that introduced language changes.
var (
	go0_0  = version{0, 0} // no version specified
	go1_9  = version{1, 9}
	go1_13 = version{1, 13}
	go1_14 = version{1, 14}
	go1_17 = version{1, 17}
	go1_18 = version{1, 18}
	go1_20 = version{1, 20}
	go1_21 = version{1, 21}
)

var errVersionSyntax = errors.New("invalid Go version syntax")

// parseGoVersion parses a Go version string (such as "go1.12")
// and returns the version, or an error. If s is the empty
// string, the version is 0.0.
func parseGoVersion(s string) (v version, err error) {
	if s == "" {
		return
	}
	if !strings.HasPrefix(s, "go") {
		return version{}, errVersionSyntax
	}
	s = s[len("go"):]
	i := 0
	for ; i < len(s) && '0' <= s[i] && s[i] <= '9'; i++ {
		if i >= 10 || i == 0 && s[i] == '0' {
			return version{}, errVersionSyntax
		}
		v.major = 10*v.major + int(s[i]) - '0'
	}
	if i > 0 && i == len(s) {
		return
	}
	if i == 0 || s[i] != '.' {
		return version{}, errVersionSyntax
	}
	s = s[i+1:]
	if s == "0" {
		// We really should not accept "go1.0",
		// but we didn't reject it from the start
		// and there are now programs that use it.
		// So accept it.
		return
	}
	i = 0
	for ; i < len(s) && '0' <= s[i] && s[i] <= '9'; i++ {
		if i >= 10 || i == 0 && s[i] == '0' {
			return version{}, errVersionSyntax
		}
		v.minor = 10*v.minor + int(s[i]) - '0'
	}
	if i > 0 && i == len(s) {
		return
	}
	return version{}, errVersionSyntax
}

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

// allowVersion reports whether the given package
// is allowed to use version major.minor.
func (check *Checker) allowVersion(pkg *Package, at positioner, v version) bool {
	// We assume that imported packages have all been checked,
	// so we only have to check for the local package.
	if pkg != check.pkg {
		return true
	}

	// If the source file declares its Go version, use that to decide.
	if check.posVers != nil {
		if src, ok := check.posVers[check.fset.File(at.Pos())]; ok && src.major >= 1 {
			return !src.before(v)
		}
	}

	// Otherwise fall back to the version in the checker.
	return check.version.equal(go0_0) || !check.version.before(v)
}

// verifyVersionf is like allowVersion but also accepts a format string and arguments
// which are used to report a version error if allowVersion returns false.
func (check *Checker) verifyVersionf(pkg *Package, at positioner, v version, format string, args ...interface{}) bool {
	if !check.allowVersion(pkg, at, v) {
		check.versionErrorf(at, v, format, args...)
		return false
	}
	return true
}
