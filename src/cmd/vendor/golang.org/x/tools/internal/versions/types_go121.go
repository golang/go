// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !go1.22
// +build !go1.22

package versions

import (
	"go/ast"
	"go/types"
)

// FileVersion returns a language version (<=1.21) derived from runtime.Version()
// or an unknown future version.
func FileVersion(info *types.Info, file *ast.File) string {
	// In x/tools built with Go <= 1.21, we do not have Info.FileVersions
	// available. We use a go version derived from the toolchain used to
	// compile the tool by default.
	// This will be <= go1.21. We take this as the maximum version that
	// this tool can support.
	//
	// There are no features currently in x/tools that need to tell fine grained
	// differences for versions <1.22.
	return toolchain
}

// InitFileVersions is a noop when compiled with this Go version.
func InitFileVersions(*types.Info) {}
