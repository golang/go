// +build go1.13

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package proxydir

import (
	"path/filepath"
	"strings"
)

// ToURL returns the file uri for a proxy directory.
func ToURL(dir string) string {
	// file URLs on Windows must start with file:///. See golang.org/issue/6027.
	path := filepath.ToSlash(dir)
	if !strings.HasPrefix(path, "/") {
		path = "/" + path
	}
	return "file://" + path
}
