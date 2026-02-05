// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package versions

// This file contains predicates for working with file versions to
// decide when a tool should consider a language feature enabled.

// named constants, to avoid misspelling
const (
	Go1_17 = "go1.17"
	Go1_18 = "go1.18"
	Go1_19 = "go1.19"
	Go1_20 = "go1.20"
	Go1_21 = "go1.21"
	Go1_22 = "go1.22"
	Go1_23 = "go1.23"
	Go1_24 = "go1.24"
	Go1_25 = "go1.25"
	Go1_26 = "go1.26"
)

// Future is an invalid unknown Go version sometime in the future.
// Do not use directly with Compare.
const Future = ""

// AtLeast reports whether the file version v comes after a Go release.
//
// Use this predicate to enable a behavior once a certain Go release
// has happened (and stays enabled in the future).
func AtLeast(v, release string) bool {
	if v == Future {
		return true // an unknown future version is always after y.
	}
	return Compare(Lang(v), Lang(release)) >= 0
}

// Before reports whether the file version v is strictly before a Go release.
//
// Use this predicate to disable a behavior once a certain Go release
// has happened (and stays enabled in the future).
func Before(v, release string) bool {
	if v == Future {
		return false // an unknown future version happens after y.
	}
	return Compare(Lang(v), Lang(release)) < 0
}
