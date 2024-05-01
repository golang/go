// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"internal/filepathlite"
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

func dirname(path string) string {
	vol := filepathlite.VolumeName(path)
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
// path limit imposed by Windows. If the path is short enough or already
// has the extended-length prefix, fixLongPath returns path unmodified.
// If the path is relative and joining it with the current working
// directory results in a path that is too long, fixLongPath returns
// the absolute path with the extended-length prefix.
//
// See https://learn.microsoft.com/en-us/windows/win32/fileio/naming-a-file#maximum-path-length-limitation
func fixLongPath(path string) string {
	if windows.CanUseLongPaths {
		return path
	}
	return addExtendedPrefix(path)
}

// addExtendedPrefix adds the extended path prefix (\\?\) to path.
func addExtendedPrefix(path string) string {
	if len(path) >= 4 {
		if path[:4] == `\??\` {
			// Already extended with \??\
			return path
		}
		if IsPathSeparator(path[0]) && IsPathSeparator(path[1]) && path[2] == '?' && IsPathSeparator(path[3]) {
			// Already extended with \\?\ or any combination of directory separators.
			return path
		}
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
	pathLength := len(path)
	if !filepathlite.IsAbs(path) {
		// If the path is relative, we need to prepend the working directory
		// plus a separator to the path before we can determine if it's too long.
		// We don't want to call syscall.Getwd here, as that call is expensive to do
		// every time fixLongPath is called with a relative path, so we use a cache.
		// Note that getwdCache might be outdated if the working directory has been
		// changed without using os.Chdir, i.e. using syscall.Chdir directly or cgo.
		// This is fine, as the worst that can happen is that we fail to fix the path.
		getwdCache.Lock()
		if getwdCache.dir == "" {
			// Init the working directory cache.
			getwdCache.dir, _ = syscall.Getwd()
		}
		pathLength += len(getwdCache.dir) + 1
		getwdCache.Unlock()
	}

	if pathLength < 248 {
		// Don't fix. (This is how Go 1.7 and earlier worked,
		// not automatically generating the \\?\ form)
		return path
	}

	var isUNC, isDevice bool
	if len(path) >= 2 && IsPathSeparator(path[0]) && IsPathSeparator(path[1]) {
		if len(path) >= 4 && path[2] == '.' && IsPathSeparator(path[3]) {
			// Starts with //./
			isDevice = true
		} else {
			// Starts with //
			isUNC = true
		}
	}
	var prefix []uint16
	if isUNC {
		// UNC path, prepend the \\?\UNC\ prefix.
		prefix = []uint16{'\\', '\\', '?', '\\', 'U', 'N', 'C', '\\'}
	} else if isDevice {
		// Don't add the extended prefix to device paths, as it would
		// change its meaning.
	} else {
		prefix = []uint16{'\\', '\\', '?', '\\'}
	}

	p, err := syscall.UTF16FromString(path)
	if err != nil {
		return path
	}
	// Estimate the required buffer size using the path length plus the null terminator.
	// pathLength includes the working directory. This should be accurate unless
	// the working directory has changed without using os.Chdir.
	n := uint32(pathLength) + 1
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
