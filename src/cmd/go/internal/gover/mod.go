// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gover

import (
	"golang.org/x/mod/module"
	"golang.org/x/mod/semver"

	"slices"
	"strings"
)

// IsToolchain reports whether the module path corresponds to the
// virtual, non-downloadable module tracking go or toolchain directives in the go.mod file.
//
// Note that IsToolchain only matches "go" and "toolchain", not the
// real, downloadable module "golang.org/toolchain" containing toolchain files.
//
//	IsToolchain("go") = true
//	IsToolchain("toolchain") = true
//	IsToolchain("golang.org/x/tools") = false
//	IsToolchain("golang.org/toolchain") = false
func IsToolchain(path string) bool {
	return path == "go" || path == "toolchain"
}

// ModCompare returns the result of comparing the versions x and y
// for the module with the given path.
// The path is necessary because the "go" and "toolchain" modules
// use a different version syntax and semantics (gover, this package)
// than most modules (semver).
func ModCompare(path string, x, y string) int {
	if path == "go" {
		return Compare(x, y)
	}
	if path == "toolchain" {
		return Compare(maybeToolchainVersion(x), maybeToolchainVersion(y))
	}
	return semver.Compare(x, y)
}

// ModSort is like module.Sort but understands the "go" and "toolchain"
// modules and their version ordering.
func ModSort(list []module.Version) {
	slices.SortFunc(list, func(a, b module.Version) int {
		if r := strings.Compare(a.Path, b.Path); r != 0 {
			return r
		}
		// To help go.sum formatting, allow version/file.
		// Compare semver prefix by semver rules,
		// file by string order.
		vi := a.Version
		vj := b.Version
		var fi, fj string
		if k := strings.Index(vi, "/"); k >= 0 {
			vi, fi = vi[:k], vi[k:]
		}
		if k := strings.Index(vj, "/"); k >= 0 {
			vj, fj = vj[:k], vj[k:]
		}
		if r := strings.Compare(vi, vj); r != 0 {
			return ModCompare(a.Path, vi, vj)
		}
		return strings.Compare(fi, fj)
	})
}

// ModIsValid reports whether vers is a valid version syntax for the module with the given path.
func ModIsValid(path, vers string) bool {
	if IsToolchain(path) {
		if path == "toolchain" {
			return IsValid(FromToolchain(vers))
		}
		return IsValid(vers)
	}
	return semver.IsValid(vers)
}

// ModIsPrefix reports whether v is a valid version syntax prefix for the module with the given path.
// The caller is assumed to have checked that ModIsValid(path, vers) is true.
func ModIsPrefix(path, vers string) bool {
	if IsToolchain(path) {
		if path == "toolchain" {
			return IsLang(FromToolchain(vers))
		}
		return IsLang(vers)
	}
	// Semver
	dots := 0
	for i := 0; i < len(vers); i++ {
		switch vers[i] {
		case '-', '+':
			return false
		case '.':
			dots++
			if dots >= 2 {
				return false
			}
		}
	}
	return true
}

// ModIsPrerelease reports whether v is a prerelease version for the module with the given path.
// The caller is assumed to have checked that ModIsValid(path, vers) is true.
func ModIsPrerelease(path, vers string) bool {
	if IsToolchain(path) {
		return IsPrerelease(vers)
	}
	return semver.Prerelease(vers) != ""
}

// ModMajorMinor returns the "major.minor" truncation of the version v,
// for use as a prefix in "@patch" queries.
func ModMajorMinor(path, vers string) string {
	if IsToolchain(path) {
		if path == "toolchain" {
			return "go" + Lang(FromToolchain(vers))
		}
		return Lang(vers)
	}
	return semver.MajorMinor(vers)
}
