// +build go1.13

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package imports

import (
	"path/filepath"
	"strings"
)

// TODO: use proxy functionality in golang.org/x/tools/go/packages/packagestest
// instead of copying it here.

func proxyDirToURL(dir string) string {
	// file URLs on Windows must start with file:///. See golang.org/issue/6027.
	path := filepath.ToSlash(dir)
	if !strings.HasPrefix(path, "/") {
		path = "/" + path
	}
	return "file://" + path
}
