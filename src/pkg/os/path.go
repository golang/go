// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import "io"

// MkdirAll creates a directory named path,
// along with any necessary parents, and returns nil,
// or else returns an error.
// The permission bits perm are used for all
// directories that MkdirAll creates.
// If path is already a directory, MkdirAll does nothing
// and returns nil.
func MkdirAll(path string, perm uint32) error {
	// If path exists, stop with success or error.
	dir, err := Stat(path)
	if err == nil {
		if dir.IsDirectory() {
			return nil
		}
		return &PathError{"mkdir", path, ENOTDIR}
	}

	// Doesn't already exist; make sure parent does.
	i := len(path)
	for i > 0 && IsPathSeparator(path[i-1]) { // Skip trailing path separator.
		i--
	}

	j := i
	for j > 0 && !IsPathSeparator(path[j-1]) { // Scan backward over element.
		j--
	}

	if j > 1 {
		// Create parent
		err = MkdirAll(path[0:j-1], perm)
		if err != nil {
			return err
		}
	}

	// Now parent exists, try to create.
	err = Mkdir(path, perm)
	if err != nil {
		// Handle arguments like "foo/." by
		// double-checking that directory doesn't exist.
		dir, err1 := Lstat(path)
		if err1 == nil && dir.IsDirectory() {
			return nil
		}
		return err
	}
	return nil
}

// RemoveAll removes path and any children it contains.
// It removes everything it can but returns the first error
// it encounters.  If the path does not exist, RemoveAll
// returns nil (no error).
func RemoveAll(path string) error {
	// Simple case: if Remove works, we're done.
	err := Remove(path)
	if err == nil {
		return nil
	}

	// Otherwise, is this a directory we need to recurse into?
	dir, serr := Lstat(path)
	if serr != nil {
		if serr, ok := serr.(*PathError); ok && (serr.Err == ENOENT || serr.Err == ENOTDIR) {
			return nil
		}
		return serr
	}
	if !dir.IsDirectory() {
		// Not a directory; return the error from Remove.
		return err
	}

	// Directory.
	fd, err := Open(path)
	if err != nil {
		return err
	}

	// Remove contents & return first error.
	err = nil
	for {
		names, err1 := fd.Readdirnames(100)
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
	}

	// Close directory, because windows won't remove opened directory.
	fd.Close()

	// Remove directory.
	err1 := Remove(path)
	if err == nil {
		err = err1
	}
	return err
}
