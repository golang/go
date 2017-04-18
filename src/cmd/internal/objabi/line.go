// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package objabi

import (
	"os"
	"path/filepath"
)

// WorkingDir returns the current working directory
// (or "/???" if the directory cannot be identified),
// with "/" as separator.
func WorkingDir() string {
	var path string
	path, _ = os.Getwd()
	if path == "" {
		path = "/???"
	}
	return filepath.ToSlash(path)
}

// AbsFile returns the absolute filename for file in the given directory.
// It also removes a leading pathPrefix, or else rewrites a leading $GOROOT
// prefix to the literal "$GOROOT".
// If the resulting path is the empty string, the result is "??".
func AbsFile(dir, file, pathPrefix string) string {
	abs := file
	if dir != "" && !filepath.IsAbs(file) {
		abs = filepath.Join(dir, file)
	}

	if pathPrefix != "" && hasPathPrefix(abs, pathPrefix) {
		if abs == pathPrefix {
			abs = ""
		} else {
			abs = abs[len(pathPrefix)+1:]
		}
	} else if hasPathPrefix(abs, GOROOT) {
		abs = "$GOROOT" + abs[len(GOROOT):]
	}
	if abs == "" {
		abs = "??"
	}

	return filepath.Clean(abs)
}

// Does s have t as a path prefix?
// That is, does s == t or does s begin with t followed by a slash?
// For portability, we allow ASCII case folding, so that hasPathPrefix("a/b/c", "A/B") is true.
// Similarly, we allow slash folding, so that hasPathPrefix("a/b/c", "a\\b") is true.
// We do not allow full Unicode case folding, for fear of causing more confusion
// or harm than good. (For an example of the kinds of things that can go wrong,
// see http://article.gmane.org/gmane.linux.kernel/1853266.)
func hasPathPrefix(s string, t string) bool {
	if len(t) > len(s) {
		return false
	}
	var i int
	for i = 0; i < len(t); i++ {
		cs := int(s[i])
		ct := int(t[i])
		if 'A' <= cs && cs <= 'Z' {
			cs += 'a' - 'A'
		}
		if 'A' <= ct && ct <= 'Z' {
			ct += 'a' - 'A'
		}
		if cs == '\\' {
			cs = '/'
		}
		if ct == '\\' {
			ct = '/'
		}
		if cs != ct {
			return false
		}
	}
	return i >= len(s) || s[i] == '/' || s[i] == '\\'
}
