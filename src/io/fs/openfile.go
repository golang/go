// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fs

import "syscall"

const (
	O_RDONLY int = syscall.O_RDONLY // open the file read-only.
	O_WRONLY int = syscall.O_WRONLY // open the file write-only.
	O_RDWR   int = syscall.O_RDWR   // open the file read-write.
	O_APPEND int = syscall.O_APPEND // append data to the file when writing.
	O_CREATE int = syscall.O_CREAT  // create a new file if none exists.
	O_EXCL   int = syscall.O_EXCL   // used with O_CREATE, file must not exist.
	O_SYNC   int = syscall.O_SYNC   // open for synchronous I/O.
	O_TRUNC  int = syscall.O_TRUNC  // truncate regular writable file when opened.
)

// WriterFile is an interface that combines the File interface with the
// ability to write data and truncate the file.
type WriterFile interface {
	File

	// Write writes len(b) bytes from b to the File.
	// It returns the number of bytes written and an error, if any.
	// Write returns a non-nil error when n != len(b).
	Write([]byte) (int, error)
}

// OpenFileFS is the interface implemented by a file system
// that provides an OpenFile method that allows opening a file
// with flags and permissions.
type OpenFileFS interface {
	FS

	// Create creates or truncates the named file. If the file already exists,
	// it is truncated. If the file does not exist, it is created with mode 0o666.
	// If there is an error, it will be of type *PathError.
	Create(name string) (WriterFile, error)

	// OpenFile is the generalized open call; most users will use Open
	// or Create instead. It opens the named file with specified flag
	// ([O_RDONLY] etc.). If the file does not exist, and the [O_CREATE] flag
	// is passed, it is created with mode perm (before umask);
	// the containing directory must exist. If successful,
	// methods on the returned [WriterFile] can be used for I/O.
	// If there is an error, it will be of type *PathError.
	OpenFile(name string, flag int, perm FileMode) (WriterFile, error)

	// Rename renames (moves) oldpath to newpath.
	// If there is an error, it will be of type [*LinkError].
	Rename(oldname, newname string) error
}

// Create attempts to create a new file with the specified name in the provided file system (fsys).
// It returns a WriterFile interface for the created file and an error, if any.
// If the fsys does not implement the OpenFileFS interface, Create returns a PathError with ErrInvalid.
func Create(fsys FS, name string) (WriterFile, error) {
	sym, ok := fsys.(OpenFileFS)
	if !ok {
		return nil, &PathError{Op: "openfile", Path: name, Err: ErrInvalid}
	}
	return sym.Create(name)
}

// OpenFile attempts to open the named file within the provided file system (fsys)
// with the specified flag and permission (perm). The fsys must implement the
// OpenFileFS interface; otherwise, OpenFile returns a PathError with ErrInvalid.
// On success, it returns a WriterFile for the opened file, or an error if the
// operation fails.
func OpenFile(fsys FS, name string, flag int, perm FileMode) (WriterFile, error) {
	sym, ok := fsys.(OpenFileFS)
	if !ok {
		return nil, &PathError{Op: "openfile", Path: name, Err: ErrInvalid}
	}
	return sym.OpenFile(name, flag, perm)
}


// Rename attempts to rename a file or directory from oldpath to newpath within the given FS.
// If the provided FS does not implement the PropertiesFS interface, it returns a PathError with ErrInvalid.
// Otherwise, it delegates the rename operation to the underlying PropertiesFS implementation.
func Rename(fsys FS, oldpath, newpath string) error {
	sym, ok := fsys.(OpenFileFS)
	if !ok {
		return &PathError{Op: "rename", Path: oldpath, Err: ErrInvalid}
	}
	return sym.Rename(oldpath, newpath)
}