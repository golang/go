// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fsys

import (
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
)

// Copied from path/filepath.

// Glob is like filepath.Glob but uses the overlay file system.
func Glob(pattern string) (matches []string, err error) {
	Trace("Glob", pattern)
	// Check pattern is well-formed.
	if _, err := filepath.Match(pattern, ""); err != nil {
		return nil, err
	}
	if !hasMeta(pattern) {
		if _, err = Lstat(pattern); err != nil {
			return nil, nil
		}
		return []string{pattern}, nil
	}

	dir, file := filepath.Split(pattern)
	volumeLen := 0
	if runtime.GOOS == "windows" {
		volumeLen, dir = cleanGlobPathWindows(dir)
	} else {
		dir = cleanGlobPath(dir)
	}

	if !hasMeta(dir[volumeLen:]) {
		return glob(dir, file, nil)
	}

	// Prevent infinite recursion. See issue 15879.
	if dir == pattern {
		return nil, filepath.ErrBadPattern
	}

	var m []string
	m, err = Glob(dir)
	if err != nil {
		return
	}
	for _, d := range m {
		matches, err = glob(d, file, matches)
		if err != nil {
			return
		}
	}
	return
}

// cleanGlobPath prepares path for glob matching.
func cleanGlobPath(path string) string {
	switch path {
	case "":
		return "."
	case string(filepath.Separator):
		// do nothing to the path
		return path
	default:
		return path[0 : len(path)-1] // chop off trailing separator
	}
}

func volumeNameLen(path string) int {
	isSlash := func(c uint8) bool {
		return c == '\\' || c == '/'
	}
	if len(path) < 2 {
		return 0
	}
	// with drive letter
	c := path[0]
	if path[1] == ':' && ('a' <= c && c <= 'z' || 'A' <= c && c <= 'Z') {
		return 2
	}
	// is it UNC? https://learn.microsoft.com/en-us/windows/win32/fileio/naming-a-file
	if l := len(path); l >= 5 && isSlash(path[0]) && isSlash(path[1]) &&
		!isSlash(path[2]) && path[2] != '.' {
		// first, leading `\\` and next shouldn't be `\`. its server name.
		for n := 3; n < l-1; n++ {
			// second, next '\' shouldn't be repeated.
			if isSlash(path[n]) {
				n++
				// third, following something characters. its share name.
				if !isSlash(path[n]) {
					if path[n] == '.' {
						break
					}
					for ; n < l; n++ {
						if isSlash(path[n]) {
							break
						}
					}
					return n
				}
				break
			}
		}
	}
	return 0
}

// cleanGlobPathWindows is windows version of cleanGlobPath.
func cleanGlobPathWindows(path string) (prefixLen int, cleaned string) {
	vollen := volumeNameLen(path)
	switch {
	case path == "":
		return 0, "."
	case vollen+1 == len(path) && os.IsPathSeparator(path[len(path)-1]): // /, \, C:\ and C:/
		// do nothing to the path
		return vollen + 1, path
	case vollen == len(path) && len(path) == 2: // C:
		return vollen, path + "." // convert C: into C:.
	default:
		if vollen >= len(path) {
			vollen = len(path) - 1
		}
		return vollen, path[0 : len(path)-1] // chop off trailing separator
	}
}

// glob searches for files matching pattern in the directory dir
// and appends them to matches. If the directory cannot be
// opened, it returns the existing matches. New matches are
// added in lexicographical order.
func glob(dir, pattern string, matches []string) (m []string, e error) {
	m = matches
	fi, err := Stat(dir)
	if err != nil {
		return // ignore I/O error
	}
	if !fi.IsDir() {
		return // ignore I/O error
	}

	list, err := ReadDir(dir)
	if err != nil {
		return // ignore I/O error
	}

	names := make([]string, 0, len(list))
	for _, info := range list {
		names = append(names, info.Name())
	}
	slices.Sort(names)

	for _, n := range names {
		matched, err := filepath.Match(pattern, n)
		if err != nil {
			return m, err
		}
		if matched {
			m = append(m, filepath.Join(dir, n))
		}
	}
	return
}

// hasMeta reports whether path contains any of the magic characters
// recognized by filepath.Match.
func hasMeta(path string) bool {
	magicChars := `*?[`
	if runtime.GOOS != "windows" {
		magicChars = `*?[\`
	}
	return strings.ContainsAny(path, magicChars)
}
