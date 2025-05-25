// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fs

// RemoveFS is the interface implemented that supporter
// to remove Folder or File.
type RemoveFS interface {
	FS

	// Remove removes the named file or (empty) directory.
	// If there is an error, it will be of type [*PathError].
	Remove(name string) error

	// RemoveAll removes path and any children it contains.
	// It removes everything it can but returns the first error
	// it encounters. If the path does not exist, RemoveAll
	// returns nil (no error).
	// If there is an error, it will be of type [*PathError].
	RemoveAll(name string) error
}

// Remove attempts to remove the file or directory specified by 'name' from the provided file system 'fsys'.
// It checks if 'fsys' implements the RemoveFS interface. If not, it returns a PathError with ErrInvalid.
// If 'fsys' supports removal, it delegates the operation to fsys.Remove(name).
func Remove(fsys FS, name string) error {
	sym, ok := fsys.(RemoveFS)
	if !ok {
		return &PathError{Op: "remove", Path: name, Err: ErrInvalid}
	}
	return sym.Remove(name)
}

// RemoveAll attempts to remove the named file or directory and any children it contains
// from the provided file system (fsys). It first checks if fsys implements the RemoveFS
// interface. If not, it returns a PathError with ErrInvalid. If fsys does implement
// RemoveFS, it delegates the removal operation to fsys.RemoveAll. The function returns
// any error encountered during the removal process.
func RemoveAll(fsys FS, name string) error {
	sym, ok := fsys.(RemoveFS)
	if !ok {
		return &PathError{Op: "remove", Path: name, Err: ErrInvalid}
	}
	return sym.RemoveAll(name)
}
