// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.12

package testenv

import "runtime/debug"

func packageMainIsDevelModule() bool {
	info, ok := debug.ReadBuildInfo()
	if !ok {
		// Most test binaries currently lack build info, but this should become more
		// permissive once https://golang.org/issue/33976 is fixed.
		return true
	}

	// Note: info.Main.Version describes the version of the module containing
	// package main, not the version of “the main module”.
	// See https://golang.org/issue/33975.
	return info.Main.Version == "(devel)"
}

func init() {
	packageMainIsDevel = packageMainIsDevelModule
}
