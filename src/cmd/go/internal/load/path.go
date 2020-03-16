// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package load

import (
	"path/filepath"
)

// expandPath returns the symlink-expanded form of path.
func expandPath(p string) string {
	x, err := filepath.EvalSymlinks(p)
	if err == nil {
		return x
	}
	return p
}
