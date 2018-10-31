// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build aix darwin dragonfly freebsd js,wasm linux nacl netbsd openbsd solaris

package os

const (
	PathSeparator     = '/' // OS-specific path separator
	PathListSeparator = ':' // OS-specific path list separator
)

// IsPathSeparator reports whether c is a directory separator character.
func IsPathSeparator(c uint8) bool {
	return PathSeparator == c
}

// basename removes trailing slashes and the leading directory name from path name.
func basename(name string) string {
	i := len(name) - 1
	// Remove trailing slashes
	for ; i > 0 && name[i] == '/'; i-- {
		name = name[:i]
	}
	// Remove leading directory name
	for i--; i >= 0; i-- {
		if name[i] == '/' {
			name = name[i+1:]
			break
		}
	}

	return name
}

// splitPath returns the base name and parent directory.
func splitPath(path string) (string, string) {
	// if no better parent is found, the path is relative from "here"
	dirname := "."
	// if no slashes in path, base is path
	basename := path

	i := len(path) - 1

	// Remove trailing slashes
	for ; i > 0 && path[i] == '/'; i-- {
		path = path[:i]
	}

	// Remove leading directory path
	for i--; i >= 0; i-- {
		if path[i] == '/' {
			dirname = path[:i+1]
			basename = path[i+1:]
			break
		}
	}

	return dirname, basename
}

func fixRootDirectory(p string) string {
	return p
}
