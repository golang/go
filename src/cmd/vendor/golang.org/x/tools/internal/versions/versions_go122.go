// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.22
// +build go1.22

package versions

import (
	"go/version"
)

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
func Lang(x string) string { return version.Lang(x) }

// Compare returns -1, 0, or +1 depending on whether
// x < y, x == y, or x > y, interpreted as Go versions.
// The versions x and y must begin with a "go" prefix: "go1.21" not "1.21".
// Invalid versions, including the empty string, compare less than
// valid versions and equal to each other.
// The language version "go1.21" compares less than the
// release candidate and eventual releases "go1.21rc1" and "go1.21.0".
// Custom toolchain suffixes are ignored during comparison:
// "go1.21.0" and "go1.21.0-bigcorp" are equal.
func Compare(x, y string) int { return version.Compare(x, y) }

// IsValid reports whether the version x is valid.
func IsValid(x string) bool { return version.IsValid(x) }
