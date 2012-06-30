// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package filepath

import "strings"

// IsAbs returns true if the path is absolute.
func IsAbs(path string) bool {
	return strings.HasPrefix(path, "/") || strings.HasPrefix(path, "#")
}

// volumeNameLen returns length of the leading volume name on Windows.
// It returns 0 elsewhere.
func volumeNameLen(path string) int {
	return 0
}

// HasPrefix exists for historical compatibility and should not be used.
func HasPrefix(p, prefix string) bool {
	return strings.HasPrefix(p, prefix)
}
