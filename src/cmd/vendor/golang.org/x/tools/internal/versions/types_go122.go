// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.22
// +build go1.22

package versions

import (
	"go/ast"
	"go/types"
)

// FileVersion returns a file's Go version.
// The reported version is an unknown Future version if a
// version cannot be determined.
func FileVersion(info *types.Info, file *ast.File) string {
	// In tools built with Go >= 1.22, the Go version of a file
	// follow a cascades of sources:
	// 1) types.Info.FileVersion, which follows the cascade:
	//   1.a) file version (ast.File.GoVersion),
	//   1.b) the package version (types.Config.GoVersion), or
	// 2) is some unknown Future version.
	//
	// File versions require a valid package version to be provided to types
	// in Config.GoVersion. Config.GoVersion is either from the package's module
	// or the toolchain (go run). This value should be provided by go/packages
	// or unitchecker.Config.GoVersion.
	if v := info.FileVersions[file]; IsValid(v) {
		return v
	}
	// Note: we could instead return runtime.Version() [if valid].
	// This would act as a max version on what a tool can support.
	return Future
}

// InitFileVersions initializes info to record Go versions for Go files.
func InitFileVersions(info *types.Info) {
	info.FileVersions = make(map[*ast.File]string)
}
