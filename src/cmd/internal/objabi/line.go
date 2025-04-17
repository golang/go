// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package objabi

import (
	"internal/buildcfg"
	"os"
	"path/filepath"
	"runtime"
	"strings"
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

// AbsFile returns the absolute filename for file in the given directory,
// as rewritten by the rewrites argument.
// For unrewritten paths, AbsFile rewrites a leading $GOROOT prefix to the literal "$GOROOT".
// If the resulting path is the empty string, the result is "??".
//
// The rewrites argument is a ;-separated list of rewrites.
// Each rewrite is of the form "prefix" or "prefix=>replace",
// where prefix must match a leading sequence of path elements
// and is either removed entirely or replaced by the replacement.
func AbsFile(dir, file, rewrites string) string {
	abs := file
	if dir != "" && !filepath.IsAbs(file) {
		abs = filepath.Join(dir, file)
	}

	abs, rewritten := ApplyRewrites(abs, rewrites)
	if !rewritten && buildcfg.GOROOT != "" && hasPathPrefix(abs, buildcfg.GOROOT) {
		abs = "$GOROOT" + abs[len(buildcfg.GOROOT):]
	}

	// Rewrite paths to match the slash convention of the target.
	// This helps ensure that cross-compiled distributions remain
	// bit-for-bit identical to natively compiled distributions.
	if runtime.GOOS == "windows" {
		abs = strings.ReplaceAll(abs, `\`, "/")
	}

	if abs == "" {
		abs = "??"
	}
	return abs
}

// ApplyRewrites returns the filename for file in the given directory,
// as rewritten by the rewrites argument.
//
// The rewrites argument is a ;-separated list of rewrites.
// Each rewrite is of the form "prefix" or "prefix=>replace",
// where prefix must match a leading sequence of path elements
// and is either removed entirely or replaced by the replacement.
func ApplyRewrites(file, rewrites string) (string, bool) {
	start := 0
	for i := 0; i <= len(rewrites); i++ {
		if i == len(rewrites) || rewrites[i] == ';' {
			if new, ok := applyRewrite(file, rewrites[start:i]); ok {
				return new, true
			}
			start = i + 1
		}
	}

	return file, false
}

// applyRewrite applies the rewrite to the path,
// returning the rewritten path and a boolean
// indicating whether the rewrite applied at all.
func applyRewrite(path, rewrite string) (string, bool) {
	prefix, replace := rewrite, ""
	if j := strings.LastIndex(rewrite, "=>"); j >= 0 {
		prefix, replace = rewrite[:j], rewrite[j+len("=>"):]
	}

	if prefix == "" || !hasPathPrefix(path, prefix) {
		return path, false
	}
	if len(path) == len(prefix) {
		return replace, true
	}
	if replace == "" {
		return path[len(prefix)+1:], true
	}
	return replace + path[len(prefix):], true
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
