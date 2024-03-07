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

// FileVersions maps a file to the file's semantic Go version.
// The reported version is the zero version if a version cannot be determined.
func FileVersions(info *types.Info, file *ast.File) string {
	return info.FileVersions[file]
}

// InitFileVersions initializes info to record Go versions for Go files.
func InitFileVersions(info *types.Info) {
	info.FileVersions = make(map[*ast.File]string)
}
