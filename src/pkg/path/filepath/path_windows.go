// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package filepath

import "strings"

// IsAbs returns true if the path is absolute.
func IsAbs(path string) (b bool) {
	v := VolumeName(path)
	if v == "" {
		return false
	}
	path = path[len(v):]
	if path == "" {
		return false
	}
	return path[0] == '/' || path[0] == '\\'
}

// VolumeName returns leading volume name.  
// Given "C:\foo\bar" it returns "C:" under windows.
// On other platforms it returns "".
func VolumeName(path string) (v string) {
	if len(path) < 2 {
		return ""
	}
	// with drive letter
	c := path[0]
	if path[1] == ':' &&
		('0' <= c && c <= '9' || 'a' <= c && c <= 'z' ||
			'A' <= c && c <= 'Z') {
		return path[:2]
	}
	return ""
}

// HasPrefix tests whether the path p begins with prefix.
// It ignores case while comparing.
func HasPrefix(p, prefix string) bool {
	if strings.HasPrefix(p, prefix) {
		return true
	}
	return strings.HasPrefix(strings.ToLower(p), strings.ToLower(prefix))
}
