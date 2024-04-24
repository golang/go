// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package filepathlite manipulates operating-system file paths.
package filepathlite

import (
	"errors"
	"io/fs"
)

var errInvalidPath = errors.New("invalid path")

// Localize is filepath.Localize.
//
// It is implemented in this package to avoid a dependency cycle
// between os and file/filepath.
//
// Tests for this function are in path/filepath.
func Localize(path string) (string, error) {
	if !fs.ValidPath(path) {
		return "", errInvalidPath
	}
	return localize(path)
}
