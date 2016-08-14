// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

const (
	PathSeparator     = '\\' // OS-specific path separator
	PathListSeparator = ';'  // OS-specific path list separator
)

// IsPathSeparator reports whether c is a directory separator character.
func IsPathSeparator(c uint8) bool {
	// NOTE: Windows accept / as path separator.
	return c == '\\' || c == '/'
}

// basename removes trailing slashes and the leading
// directory name and drive letter from path name.
func basename(name string) string {
	// Remove drive letter
	if len(name) == 2 && name[1] == ':' {
		name = "."
	} else if len(name) > 2 && name[1] == ':' {
		name = name[2:]
	}
	i := len(name) - 1
	// Remove trailing slashes
	for ; i > 0 && (name[i] == '/' || name[i] == '\\'); i-- {
		name = name[:i]
	}
	// Remove leading directory name
	for i--; i >= 0; i-- {
		if name[i] == '/' || name[i] == '\\' {
			name = name[i+1:]
			break
		}
	}
	return name
}

func isAbs(path string) (b bool) {
	v := volumeName(path)
	if v == "" {
		return false
	}
	path = path[len(v):]
	if path == "" {
		return false
	}
	return IsPathSeparator(path[0])
}

func volumeName(path string) (v string) {
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
	// is it UNC
	if l := len(path); l >= 5 && IsPathSeparator(path[0]) && IsPathSeparator(path[1]) &&
		!IsPathSeparator(path[2]) && path[2] != '.' {
		// first, leading `\\` and next shouldn't be `\`. its server name.
		for n := 3; n < l-1; n++ {
			// second, next '\' shouldn't be repeated.
			if IsPathSeparator(path[n]) {
				n++
				// third, following something characters. its share name.
				if !IsPathSeparator(path[n]) {
					if path[n] == '.' {
						break
					}
					for ; n < l; n++ {
						if IsPathSeparator(path[n]) {
							break
						}
					}
					return path[:n]
				}
				break
			}
		}
	}
	return ""
}

func fromSlash(path string) string {
	// Replace each '/' with '\\' if present
	var pathbuf []byte
	var lastSlash int
	for i, b := range path {
		if b == '/' {
			if pathbuf == nil {
				pathbuf = make([]byte, len(path))
			}
			copy(pathbuf[lastSlash:], path[lastSlash:i])
			pathbuf[i] = '\\'
			lastSlash = i + 1
		}
	}
	if pathbuf == nil {
		return path
	}

	copy(pathbuf[lastSlash:], path[lastSlash:])
	return string(pathbuf)
}

func dirname(path string) string {
	vol := volumeName(path)
	i := len(path) - 1
	for i >= len(vol) && !IsPathSeparator(path[i]) {
		i--
	}
	dir := path[len(vol) : i+1]
	last := len(dir) - 1
	if last > 0 && IsPathSeparator(dir[last]) {
		dir = dir[:last]
	}
	if dir == "" {
		dir = "."
	}
	return vol + dir
}
