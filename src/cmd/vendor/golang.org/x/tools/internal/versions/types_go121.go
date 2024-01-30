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

// FileVersions always reports the a file's Go version as the
// zero version at this Go version.
func FileVersions(info *types.Info, file *ast.File) string { return "" }

// InitFileVersions is a noop at this Go version.
func InitFileVersions(*types.Info) {}
