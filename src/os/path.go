// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"io"
	"syscall"
)

// MkdirAll creates a directory named path,
// along with any necessary parents, and returns nil,
// or else returns an error.
// The permission bits perm (before umask) are used for all
// directories that MkdirAll creates.
// If path is already a directory, MkdirAll does nothing
// and returns nil.
func MkdirAll(path string, perm FileMode) error {
	// Fast path: if we can tell whether path is a directory or file, stop with success or error.
	dir, err := Stat(path)
	if err == nil {
		if dir.IsDir() {
			return nil
		}
		return &PathError{"mkdir", path, syscall.ENOTDIR}
	}

	// Slow path: make sure parent exists and then call Mkdir for path.
	i := len(path)
	for i > 0 && IsPathSeparator(path[i-1]) { // Skip trailing path separator.
		i--
	}

	j := i
	for j > 0 && !IsPathSeparator(path[j-1]) { // Scan backward over element.
		j--
	}

	if j > 1 {
		// Create parent.
		err = MkdirAll(fixRootDirectory(path[:j-1]), perm)
		if err != nil {
			return err
		}
	}

	// Parent now exists; invoke Mkdir and use its result.
	err = Mkdir(path, perm)
	if err != nil {
		// Handle arguments like "foo/." by
		// double-checking that directory doesn't exist.
		dir, err1 := Lstat(path)
		if err1 == nil && dir.IsDir() {
			return nil
		}
		return err
	}
	return nil
}

// RemoveAll removes path and any children it contains.
// It removes everything it can but returns the first error
// it encounters. If the path does not exist, RemoveAll
// returns nil (no error).
func RemoveAll(path string) error {
	// Simple case: if Remove works, we're done.
	err := Remove(path)
	if err == nil || IsNotExist(err) {
		return nil
	}

	// Otherwise, is this a directory we need to recurse into?
	dir, serr := Lstat(path)
	if serr != nil {
		if serr, ok := serr.(*PathError); ok && (IsNotExist(serr.Err) || serr.Err == syscall.ENOTDIR) {
			return nil
		}
		return serr
	}
	if !dir.IsDir() {
		// Not a directory; return the error from Remove.
		return err
	}

	// Remove contents & return first error.
	err = nil
	for {
		fd, err := Open(path)
		if err != nil {
			if IsNotExist(err) {
				// Already deleted by someone else.
				return nil
			}
			return err
		}

		const request = 1024
		names, err1 := fd.Readdirnames(request)

		// Removing files from the directory may have caused
		// the OS to reshuffle it. Simply calling Readdirnames
		// again may skip some entries. The only reliable way
		// to avoid this is to close and re-open the
		// directory. See issue 20841.
		fd.Close()

		for _, name := range names {
			err1 := RemoveAll(path + string(PathSeparator) + name)
			if err == nil {
				err = err1
			}
		}

		if err1 == io.EOF {
			break
		}
		// If Readdirnames returned an error, use it.
		if err == nil {
			err = err1
		}
		if len(names) == 0 {
			break
		}

		// We don't want to re-open unnecessarily, so if we
		// got fewer than request names from Readdirnames, try
		// simply removing the directory now. If that
		// succeeds, we are done.
		if len(names) < request {
			err1 := Remove(path)
			if err1 == nil || IsNotExist(err1) {
				return nil
			}

			if err != nil {
				// We got some error removing the
				// directory contents, and since we
				// read fewer names than we requested
				// there probably aren't more files to
				// remove. Don't loop around to read
				// the directory again. We'll probably
				// just get the same error.
				return err
			}
		}
	}

	// Remove directory.
	err1 := Remove(path)
	if err1 == nil || IsNotExist(err1) {
		return nil
	}
	if err == nil {
		err = err1
	}
	return err
}
