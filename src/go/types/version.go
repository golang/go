// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"fmt"
	"go/ast"
	"go/token"
	"regexp"
	"strconv"
	"strings"
)

// langCompat reports an error if the representation of a numeric
// literal is not compatible with the current language version.
func (check *Checker) langCompat(lit *ast.BasicLit) {
	s := lit.Value
	if len(s) <= 2 || check.allowVersion(check.pkg, 1, 13) {
		return
	}
	// len(s) > 2
	if strings.Contains(s, "_") {
		check.errorf(lit, _UnsupportedFeature, "underscores in numeric literals requires go1.13 or later")
		return
	}
	if s[0] != '0' {
		return
	}
	radix := s[1]
	if radix == 'b' || radix == 'B' {
		check.errorf(lit, _UnsupportedFeature, "binary literals requires go1.13 or later")
		return
	}
	if radix == 'o' || radix == 'O' {
		check.errorf(lit, _UnsupportedFeature, "0o/0O-style octal literals requires go1.13 or later")
		return
	}
	if lit.Kind != token.INT && (radix == 'x' || radix == 'X') {
		check.errorf(lit, _UnsupportedFeature, "hexadecimal floating-point literals requires go1.13 or later")
	}
}

// allowVersion reports whether the given package
// is allowed to use version major.minor.
func (check *Checker) allowVersion(pkg *Package, major, minor int) bool {
	// We assume that imported packages have all been checked,
	// so we only have to check for the local package.
	if pkg != check.pkg {
		return true
	}
	ma, mi := check.version.major, check.version.minor
	return ma == 0 && mi == 0 || ma > major || ma == major && mi >= minor
}

type version struct {
	major, minor int
}

// parseGoVersion parses a Go version string (such as "go1.12")
// and returns the version, or an error. If s is the empty
// string, the version is 0.0.
func parseGoVersion(s string) (v version, err error) {
	if s == "" {
		return
	}
	matches := goVersionRx.FindStringSubmatch(s)
	if matches == nil {
		err = fmt.Errorf(`should be something like "go1.12"`)
		return
	}
	v.major, err = strconv.Atoi(matches[1])
	if err != nil {
		return
	}
	v.minor, err = strconv.Atoi(matches[2])
	return
}

// goVersionRx matches a Go version string, e.g. "go1.12".
var goVersionRx = regexp.MustCompile(`^go([1-9][0-9]*)\.(0|[1-9][0-9]*)$`)
