// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"internal/syscall/windows"
	"syscall"
)

const (
	PathSeparator     = '\\' // OS-specific path separator
	PathListSeparator = ';'  // OS-specific path list separator
)

// IsPathSeparator reports whether c is a directory separator character.
func IsPathSeparator(c uint8) bool {
	// NOTE: Windows accepts / as path separator.
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

// fixLongPath returns the extended-length (\\?\-prefixed) form of
// path when needed, in order to avoid the default 260 character file
// path limit imposed by Windows. If the path is short enough or is relative,
// fixLongPath returns path unmodified.
//
// See https://learn.microsoft.com/en-us/windows/win32/fileio/naming-a-file#maximum-path-length-limitation
func fixLongPath(path string) string {
	if windows.CanUseLongPaths {
		return path
	}
	// Do nothing (and don't allocate) if the path is "short".
	// Empirically (at least on the Windows Server 2013 builder),
	// the kernel is arbitrarily okay with < 248 bytes. That
	// matches what the docs above say:
	// "When using an API to create a directory, the specified
	// path cannot be so long that you cannot append an 8.3 file
	// name (that is, the directory name cannot exceed MAX_PATH
	// minus 12)." Since MAX_PATH is 260, 260 - 12 = 248.
	//
	// The MSDN docs appear to say that a normal path that is 248 bytes long
	// will work; empirically the path must be less then 248 bytes long.
	if len(path) < 248 {
		// Don't fix. (This is how Go 1.7 and earlier worked,
		// not automatically generating the \\?\ form)
		return path
	}

	if prefix := path[:4]; prefix == `\\.\` || prefix == `\\?\` || prefix == `\??\` {
		// Don't fix. Device path or extended path form.
		return path
	}
	if !isAbs(path) {
		// Relative path
		return path
	}

	var prefix []uint16
	var isUNC bool
	if path[:2] == `\\` {
		// UNC path, prepend the \\?\UNC\ prefix.
		prefix = []uint16{'\\', '\\', '?', '\\', 'U', 'N', 'C', '\\'}
		isUNC = true
	} else {
		prefix = []uint16{'\\', '\\', '?', '\\'}
	}

	p, err := syscall.UTF16FromString(path)
	if err != nil {
		return path
	}
	n := uint32(len(p))
	var buf []uint16
	for {
		buf = make([]uint16, n+uint32(len(prefix)))
		n, err = syscall.GetFullPathName(&p[0], n, &buf[len(prefix)], nil)
		if err != nil {
			return path
		}
		if n <= uint32(len(buf)-len(prefix)) {
			buf = buf[:n+uint32(len(prefix))]
			break
		}
	}
	if isUNC {
		// Remove leading \\.
		buf = buf[2:]
	}
	copy(buf, prefix)
	return syscall.UTF16ToString(buf)
}
