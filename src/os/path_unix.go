// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix || (js && wasm) || wasip1

package os

const (
	PathSeparator     = '/' // OS-specific path separator
	PathListSeparator = ':' // OS-specific path list separator
)

// IsPathSeparator reports whether c is a directory separator character.
func IsPathSeparator(c uint8) bool {
	return PathSeparator == c
}

// splitPath returns the base name and parent directory.
func splitPath(path string) (string, string) {
	// if no better parent is found, the path is relative from "here"
	dirname := "."

	// Remove all but one leading slash.
	for len(path) > 1 && path[0] == '/' && path[1] == '/' {
		path = path[1:]
	}

	i := len(path) - 1

	// Remove trailing slashes.
	for ; i > 0 && path[i] == '/'; i-- {
		path = path[:i]
	}

	// if no slashes in path, base is path
	basename := path

	// Remove leading directory path
	for i--; i >= 0; i-- {
		if path[i] == '/' {
			if i == 0 {
				dirname = path[:1]
			} else {
				dirname = path[:i]
			}
			basename = path[i+1:]
			break
		}
	}

	return dirname, basename
}
