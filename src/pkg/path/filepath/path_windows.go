// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package filepath

import "os"

// IsAbs returns true if the path is absolute.
func IsAbs(path string) bool {
	return path != "" && (volumeName(path) != "" || os.IsPathSeparator(path[0]))
}

// volumeName return leading volume name.  
// If given "C:\foo\bar", return "C:" on windows.
func volumeName(path string) string {
	if path == "" {
		return ""
	}
	// with drive letter
	c := path[0]
	if len(path) > 2 && path[1] == ':' && os.IsPathSeparator(path[2]) &&
		('0' <= c && c <= '9' || 'a' <= c && c <= 'z' ||
			'A' <= c && c <= 'Z') {
		return path[0:2]
	}
	return ""
}
