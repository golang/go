// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fs

// SymlinkFS is the interface implemented by a file system
// that supports writer symbolic links.
type SymlinkFS interface {
	FS

	// Symlink creates newname as a symbolic link to oldname.
	// If there is an error, it will be of type [*LinkError].
	Symlink(oldname, newname string) error

	// Link creates newname as a hard link to the oldname file.
	// If there is an error, it will be of type [*LinkError].
	Link(oldname, newname string) error
}

// Symlink create a symbolic link to oldname.
//
// If fsys does not implement [SymlinkFS], then Symlink returns an error.
func Symlink(fsys FS, oldname, newname string) error {
	sym, ok := fsys.(SymlinkFS)
	if !ok {
		return &LinkError{Op: "symlink", Old: oldname, New: newname, Err: ErrInvalid}
	}
	return sym.Symlink(oldname, newname)
}

// Link creates newname as a hard link to the oldname file.
//
// If there is an error, it will be of type [*LinkError].
func Link(fsys FS, oldname, newname string) error {
	sym, ok := fsys.(SymlinkFS)
	if !ok {
		return &LinkError{Op: "link", Old: oldname, New: newname, Err: ErrInvalid}
	}
	return sym.Link(oldname, newname)
}
