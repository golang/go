// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package robustio wraps I/O functions that are prone to failure on Windows,
// transparently retrying errors up to an arbitrary timeout.
//
// Errors are classified heuristically and retries are bounded, so the functions
// in this package do not completely eliminate spurious errors. However, they do
// significantly reduce the rate of failure in practice.
//
// If so, the error will likely wrap one of:
// The functions in this package do not completely eliminate spurious errors,
// but substantially reduce their rate of occurrence in practice.
package robustio

import "time"

// Rename is like os.Rename, but on Windows retries errors that may occur if the
// file is concurrently read or overwritten.
//
// (See golang.org/issue/31247 and golang.org/issue/32188.)
func Rename(oldpath, newpath string) error {
	return rename(oldpath, newpath)
}

// ReadFile is like os.ReadFile, but on Windows retries errors that may
// occur if the file is concurrently replaced.
//
// (See golang.org/issue/31247 and golang.org/issue/32188.)
func ReadFile(filename string) ([]byte, error) {
	return readFile(filename)
}

// RemoveAll is like os.RemoveAll, but on Windows retries errors that may occur
// if an executable file in the directory has recently been executed.
//
// (See golang.org/issue/19491.)
func RemoveAll(path string) error {
	return removeAll(path)
}

// IsEphemeralError reports whether err is one of the errors that the functions
// in this package attempt to mitigate.
//
// Errors considered ephemeral include:
//   - syscall.ERROR_ACCESS_DENIED
//   - syscall.ERROR_FILE_NOT_FOUND
//   - internal/syscall/windows.ERROR_SHARING_VIOLATION
//
// This set may be expanded in the future; programs must not rely on the
// non-ephemerality of any given error.
func IsEphemeralError(err error) bool {
	return isEphemeralError(err)
}

// A FileID uniquely identifies a file in the file system.
//
// If GetFileID(name1) returns the same ID as GetFileID(name2), the two file
// names denote the same file.
// A FileID is comparable, and thus suitable for use as a map key.
type FileID struct {
	device, inode uint64
}

// GetFileID returns the file system's identifier for the file, and its
// modification time.
// Like os.Stat, it reads through symbolic links.
func GetFileID(filename string) (FileID, time.Time, error) { return getFileID(filename) }
