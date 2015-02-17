// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package filepath

import "strings"

// IsAbs reports whether the path is absolute.
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

func splitList(path string) []string {
	if path == "" {
		return []string{}
	}
	return strings.Split(path, string(ListSeparator))
}

func abs(path string) (string, error) {
	return unixAbs(path)
}

func join(elem []string) string {
	// If there's a bug here, fix the logic in ./path_unix.go too.
	for i, e := range elem {
		if e != "" {
			return Clean(strings.Join(elem[i:], string(Separator)))
		}
	}
	return ""
}
