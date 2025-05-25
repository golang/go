// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fs

// MkdirFS is an interface that extends the FS interface and provides
// methods for creating directories within a file system.
type MkdirFS interface {
	FS

	// Mkdir creates a new directory with the specified name and permission
	// bits (before umask).
	// If there is an error, it will be of type *PathError.
	Mkdir(name string, perm FileMode) error

	// MkdirAll creates a directory named path,
	// along with any necessary parents, and returns nil,
	// or else returns an error.
	// The permission bits perm (before umask) are used for all
	// directories that MkdirAll creates.
	// If path is already a directory, MkdirAll does nothing
	// and returns nil.
	MkdirAll(name string, perm FileMode) error
}

// Mkdir creates a new directory with the specified name and permission bits,
// relative to the provided file system.
//
// If fsys does not implement MkdirFS, Mkdir returns an error of type PathError
// with ErrInvalid.
func Mkdir(fsys FS, name string, perm FileMode) error {
	sym, ok := fsys.(MkdirFS)
	if !ok {
		return &PathError{Op: "mkdir", Path: name, Err: ErrInvalid}
	}
	return sym.Mkdir(name, perm)
}

// MkdirAll creates a new directory with the specified name and permission bits,
// along with any necessary parents. If the directory already exists, MkdirAll
// returns nil (no error).
//
// The permission bits perm are used for all directories
// that MkdirAll creates, including the final one. If fsys does not implement
// MkdirFS, MkdirAll returns an error of type PathError with Err set to ErrInvalid.
func MkdirAll(fsys FS, name string, perm FileMode) error {
	sym, ok := fsys.(MkdirFS)
	if !ok {
		return &PathError{Op: "mkdirall", Path: name, Err: ErrInvalid}
	}
	return sym.MkdirAll(name, perm)
}
