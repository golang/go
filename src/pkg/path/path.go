// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package path implements utility routines for manipulating slash-separated
// filename paths.
package path

import (
	"strings"
)

// Clean returns the shortest path name equivalent to path
// by purely lexical processing.  It applies the following rules
// iteratively until no further processing can be done:
//
//	1. Replace multiple slashes with a single slash.
//	2. Eliminate each . path name element (the current directory).
//	3. Eliminate each inner .. path name element (the parent directory)
//	   along with the non-.. element that precedes it.
//	4. Eliminate .. elements that begin a rooted path:
//	   that is, replace "/.." by "/" at the beginning of a path.
//
// If the result of this process is an empty string, Clean
// returns the string ".".
//
// See also Rob Pike, ``Lexical File Names in Plan 9 or
// Getting Dot-Dot right,''
// http://plan9.bell-labs.com/sys/doc/lexnames.html
func Clean(path string) string {
	if path == "" {
		return "."
	}

	rooted := path[0] == '/'
	n := len(path)

	// Invariants:
	//	reading from path; r is index of next byte to process.
	//	writing to buf; w is index of next byte to write.
	//	dotdot is index in buf where .. must stop, either because
	//		it is the leading slash or it is a leading ../../.. prefix.
	buf := []byte(path)
	r, w, dotdot := 0, 0, 0
	if rooted {
		r, w, dotdot = 1, 1, 1
	}

	for r < n {
		switch {
		case path[r] == '/':
			// empty path element
			r++
		case path[r] == '.' && (r+1 == n || path[r+1] == '/'):
			// . element
			r++
		case path[r] == '.' && path[r+1] == '.' && (r+2 == n || path[r+2] == '/'):
			// .. element: remove to last /
			r += 2
			switch {
			case w > dotdot:
				// can backtrack
				w--
				for w > dotdot && buf[w] != '/' {
					w--
				}
			case !rooted:
				// cannot backtrack, but not rooted, so append .. element.
				if w > 0 {
					buf[w] = '/'
					w++
				}
				buf[w] = '.'
				w++
				buf[w] = '.'
				w++
				dotdot = w
			}
		default:
			// real path element.
			// add slash if needed
			if rooted && w != 1 || !rooted && w != 0 {
				buf[w] = '/'
				w++
			}
			// copy element
			for ; r < n && path[r] != '/'; r++ {
				buf[w] = path[r]
				w++
			}
		}
	}

	// Turn empty string into "."
	if w == 0 {
		buf[w] = '.'
		w++
	}

	return string(buf[0:w])
}

// Split splits path immediately following the final path separator,
// separating it into a directory and file name component.
// If there is no separator in path, Split returns an empty dir and
// file set to path.
func Split(path string) (dir, file string) {
	i := strings.LastIndex(path, "/")
	return path[:i+1], path[i+1:]
}

// Join joins any number of path elements into a single path, adding a
// separating slash if necessary.  All empty strings are ignored.
func Join(elem ...string) string {
	for i, e := range elem {
		if e != "" {
			return Clean(strings.Join(elem[i:], "/"))
		}
	}
	return ""
}

// Ext returns the file name extension used by path.
// The extension is the suffix beginning at the final dot
// in the final slash-separated element of path;
// it is empty if there is no dot.
func Ext(path string) string {
	for i := len(path) - 1; i >= 0 && path[i] != '/'; i-- {
		if path[i] == '.' {
			return path[i:]
		}
	}
	return ""
}

// Base returns the last element of path.
// Trailing slashes are removed before extracting the last element.
// If the path is empty, Base returns ".".
// If the path consists entirely of slashes, Base returns "/".
func Base(path string) string {
	if path == "" {
		return "."
	}
	// Strip trailing slashes.
	for len(path) > 0 && path[len(path)-1] == '/' {
		path = path[0 : len(path)-1]
	}
	// Find the last element
	if i := strings.LastIndex(path, "/"); i >= 0 {
		path = path[i+1:]
	}
	// If empty now, it had only slashes.
	if path == "" {
		return "/"
	}
	return path
}

// IsAbs returns true if the path is absolute.
func IsAbs(path string) bool {
	return len(path) > 0 && path[0] == '/'
}

// Dir returns the all but the last element of path, typically the path's directory.
// Trailing path separators are removed before processing.
// If the path is empty, Dir returns ".".
// If the path consists entirely of separators, Dir returns a single separator.
// The returned path does not end in a separator unless it is the root directory.
func Dir(path string) string {
	dir, _ := Split(path)
	dir = Clean(dir)
	last := len(dir) - 1
	if last > 0 && dir[last] == '/' {
		dir = dir[:last]
	}
	if dir == "" {
		dir = "."
	}
	return dir
}
