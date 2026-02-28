// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fs

// ReadLinkFS is the interface implemented by a file system
// that supports reading symbolic links.
type ReadLinkFS interface {
	FS

	// ReadLink returns the destination of the named symbolic link.
	// If there is an error, it should be of type [*PathError].
	ReadLink(name string) (string, error)

	// Lstat returns a [FileInfo] describing the named file.
	// If the file is a symbolic link, the returned [FileInfo] describes the symbolic link.
	// Lstat makes no attempt to follow the link.
	// If there is an error, it should be of type [*PathError].
	Lstat(name string) (FileInfo, error)
}

// ReadLink returns the destination of the named symbolic link.
//
// If fsys does not implement [ReadLinkFS], then ReadLink returns an error.
func ReadLink(fsys FS, name string) (string, error) {
	sym, ok := fsys.(ReadLinkFS)
	if !ok {
		return "", &PathError{Op: "readlink", Path: name, Err: ErrInvalid}
	}
	return sym.ReadLink(name)
}

// Lstat returns a [FileInfo] describing the named file.
// If the file is a symbolic link, the returned [FileInfo] describes the symbolic link.
// Lstat makes no attempt to follow the link.
//
// If fsys does not implement [ReadLinkFS], then Lstat is identical to [Stat].
func Lstat(fsys FS, name string) (FileInfo, error) {
	sym, ok := fsys.(ReadLinkFS)
	if !ok {
		return Stat(fsys, name)
	}
	return sym.Lstat(name)
}
