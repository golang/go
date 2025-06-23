// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package telemetry

import (
	"os"
	"path/filepath"
	"runtime/debug"
	"strings"
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
	// TODO(matloob): Use go/version.IsValid instead of checking for X: once the telemetry
	// module can be upgraded to require Go 1.22.
	if strings.Contains(goVers, "devel") || strings.Contains(goVers, "-") || strings.Contains(goVers, "X:") {
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
		if strings.Contains(progVers, "devel") || strings.Count(progVers, "-") > 1 {
			// Heuristically mark all pseudo-version-like version strings as "devel"
			// to avoid creating too many counter files.
			// We should not use regexp that pulls in large dependencies.
			// Pseudo-versions have at least three parts (https://go.dev/ref/mod#pseudo-versions).
			// This heuristic still allows use to track prerelease
			// versions (e.g. gopls@v0.16.0-pre.1, vscgo@v0.42.0-rc.1).
			progVers = "devel"
		}
	}

	return goVers, progPath, progVers
}
