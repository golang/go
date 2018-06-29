// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package load

import (
	"path/filepath"
	"strings"
)

// hasSubdir reports whether dir is a subdirectory of
// (possibly multiple levels below) root.
// If so, it sets rel to the path fragment that must be
// appended to root to reach dir.
func hasSubdir(root, dir string) (rel string, ok bool) {
	if p, err := filepath.EvalSymlinks(root); err == nil {
		root = p
	}
	if p, err := filepath.EvalSymlinks(dir); err == nil {
		dir = p
	}
	const sep = string(filepath.Separator)
	root = filepath.Clean(root)
	if !strings.HasSuffix(root, sep) {
		root += sep
	}
	dir = filepath.Clean(dir)
	if !strings.HasPrefix(dir, root) {
		return "", false
	}
	return filepath.ToSlash(dir[len(root):]), true
}

// expandPath returns the symlink-expanded form of path.
func expandPath(p string) string {
	x, err := filepath.EvalSymlinks(p)
	if err == nil {
		return x
	}
	return p
}
