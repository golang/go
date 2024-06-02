// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package telemetry

import (
	"os"
	"path/filepath"
	"runtime/debug"
	"strings"

	"golang.org/x/mod/module"
)

// IsToolchainProgram reports whether a program with the given path is a Go
// toolchain program.
func IsToolchainProgram(progPath string) bool {
	return strings.HasPrefix(progPath, "cmd/")
}

// ProgramInfo extracts the go version, program package path, and program
// version to use for counter files.
//
// For programs in the Go toolchain, the program version will be the same as
// the Go version, and will typically be of the form "go1.2.3", not a semantic
// version of the form "v1.2.3". Go versions may also include spaces and
// special characters.
func ProgramInfo(info *debug.BuildInfo) (goVers, progPath, progVers string) {
	goVers = info.GoVersion
	if strings.Contains(goVers, "devel") || strings.Contains(goVers, "-") {
		goVers = "devel"
	}

	progPath = info.Path
	if progPath == "" {
		progPath = strings.TrimSuffix(filepath.Base(os.Args[0]), ".exe")
	}

	// Main module version information is not populated for the cmd module, but
	// we can re-use the Go version here.
	if IsToolchainProgram(progPath) {
		progVers = goVers
	} else {
		progVers = info.Main.Version
		if strings.Contains(progVers, "devel") || module.IsPseudoVersion(progVers) {
			// We don't want to track pseudo versions, but may want to track prereleases.
			progVers = "devel"
		}
	}

	return goVers, progPath, progVers
}
