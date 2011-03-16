// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package filepath

const (
	Separator     = '\\' // OS-specific path separator
	ListSeparator = ':'  // OS-specific path list separator
)

// isSeparator returns true if c is a directory separator character.
func isSeparator(c uint8) bool {
	// NOTE: Windows accept / as path separator.
	return c == '\\' || c == '/'
}

// IsAbs returns true if the path is absolute.
func IsAbs(path string) bool {
	return path != "" && (volumeName(path) != "" || isSeparator(path[0]))
}

// volumeName return leading volume name.  
// If given "C:\foo\bar", return "C:" on windows.
func volumeName(path string) string {
	if path == "" {
		return ""
	}
	// with drive letter
	c := path[0]
	if len(path) > 2 && path[1] == ':' && isSeparator(path[2]) &&
		('0' <= c && c <= '9' || 'a' <= c && c <= 'z' ||
			'A' <= c && c <= 'Z') {
		return path[0:2]
	}
	return ""
}
