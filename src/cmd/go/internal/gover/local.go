// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gover

import (
	"internal/goversion"
	"runtime"
	"strconv"
)

// TestVersion is initialized in the go command test binary
// to be $TESTGO_VERSION, to allow tests to override the
// go command's idea of its own version as returned by Local.
var TestVersion string

// Local returns the local Go version, the one implemented by this go command.
func Local() string {
	v, _ := local()
	return v
}

// LocalToolchain returns the local toolchain name, the one implemented by this go command.
func LocalToolchain() string {
	_, t := local()
	return t
}

func local() (goVers, toolVers string) {
	toolVers = runtime.Version()
	if TestVersion != "" {
		toolVers = TestVersion
	}
	goVers = FromToolchain(toolVers)
	if goVers == "" {
		// Development branch. Use "Dev" version with just 1.N, no rc1 or .0 suffix.
		goVers = "1." + strconv.Itoa(goversion.Version)
		toolVers = "go" + goVers
	}
	return goVers, toolVers
}
