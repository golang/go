// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package filepath

import "strings"

const (
	Separator     = '/' // OS-specific path separator
	ListSeparator = 0   // OS-specific path list separator
)

// isSeparator returns true if c is a directory separator character.
func isSeparator(c uint8) bool {
	return Separator == c
}

// IsAbs returns true if the path is absolute.
func IsAbs(path string) bool {
	return strings.HasPrefix(path, "/") || strings.HasPrefix(path, "#")
}

// volumeName returns the leading volume name on Windows.
// It returns "" elsewhere
func volumeName(path string) string {
	return ""
}
