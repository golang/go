// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fs

import "time"

// Properties is an interface for file systems that provide
// extended file properties manipulation capabilities, such as
// changing file modes, ownership, and timestamps.
//
// Implementations of this interface allow for more fine-grained
// control over file attributes beyond basic read and write operations.
type PropertiesFS interface {
	FS

	// Chmod changes the mode of the named file to mode.
	// If the file is a symbolic link, it changes the mode of the link's target.
	// If there is an error, it will be of type [*PathError].
	Chmod(name string, mode FileMode) error

	// Chown changes the numeric uid and gid of the named file.
	// If the file is a symbolic link, it changes the uid and gid of the link's target.
	// A uid or gid of -1 means to not change that value.
	// If there is an error, it will be of type [*PathError].
	Chown(name string, uid, gid int) error

	// Chtimes changes the access and modification times of the named
	// file, similar to the Unix utime() or utimes() functions.
	// A zero [time.Time] value will leave the corresponding file time unchanged.
	Chtimes(name string, atime time.Time, mtime time.Time) error
}

// Chmod changes the mode of the named file within the provided file system (fsys) to the specified FileMode.
// If the file system does not implement the PropertiesFS interface, Chmod returns a PathError with ErrInvalid.
// Otherwise, it delegates the operation to the file system's Chmod method.
func Chmod(fsys FS, name string, mode FileMode) error {
	sym, ok := fsys.(PropertiesFS)
	if !ok {
		return &PathError{Op: "chmod", Path: name, Err: ErrInvalid}
	}
	return sym.Chmod(name, mode)
}

// Chown changes the numeric user and group ownership of the named file within the provided FS.
// It attempts to cast the FS to a PropertiesFS to perform the operation. If the FS does not
// implement PropertiesFS, Chown returns a PathError with ErrInvalid.
// The uid and gid parameters specify the new user and group IDs, respectively.
func Chown(fsys FS, name string, uid, gid int) error {
	sym, ok := fsys.(PropertiesFS)
	if !ok {
		return &PathError{Op: "chown", Path: name, Err: ErrInvalid}
	}
	return sym.Chown(name, uid, gid)
}

// Chtimes sets the access and modification times of the named file within the provided file system (fsys).
// It attempts to cast fsys to a PropertiesFS interface, which must implement the Chtimes method.
// If fsys does not implement PropertiesFS, Chtimes returns a PathError with ErrInvalid.
// Otherwise, it delegates the operation to the underlying PropertiesFS implementation.
func Chtimes(fsys FS, name string, atime time.Time, mtime time.Time) error {
	sym, ok := fsys.(PropertiesFS)
	if !ok {
		return &PathError{Op: "chtimes", Path: name, Err: ErrInvalid}
	}
	return sym.Chtimes(name, atime, mtime)
}
