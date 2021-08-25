// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build gccgo

package goroot

import (
	"os"
	"path/filepath"
)

// IsStandardPackage reports whether path is a standard package,
// given goroot and compiler.
func IsStandardPackage(goroot, compiler, path string) bool {
	switch compiler {
	case "gc":
		dir := filepath.Join(goroot, "src", path)
		_, err := os.Stat(dir)
		return err == nil
	case "gccgo":
		return stdpkg[path]
	default:
		panic("unknown compiler " + compiler)
	}
}
