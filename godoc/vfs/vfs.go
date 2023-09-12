// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package vfs defines types for abstract file system access and provides an
// implementation accessing the file system of the underlying OS.
package vfs // import "golang.org/x/tools/godoc/vfs"

import (
	"io"
	"os"
)

// RootType indicates the type of files contained within a directory.
//
// It is used to indicate whether a directory is the root
// of a GOROOT, a GOPATH, or neither.
// An empty string represents the case when a directory is neither.
type RootType string

const (
	RootTypeGoRoot RootType = "GOROOT"
	RootTypeGoPath RootType = "GOPATH"
)

// The FileSystem interface specifies the methods godoc is using
// to access the file system for which it serves documentation.
type FileSystem interface {
	Opener
	Lstat(path string) (os.FileInfo, error)
	Stat(path string) (os.FileInfo, error)
	ReadDir(path string) ([]os.FileInfo, error)
	RootType(path string) RootType
	String() string
}

// Opener is a minimal virtual filesystem that can only open regular files.
type Opener interface {
	Open(name string) (ReadSeekCloser, error)
}

// A ReadSeekCloser can Read, Seek, and Close.
type ReadSeekCloser interface {
	io.Reader
	io.Seeker
	io.Closer
}

// ReadFile reads the file named by path from fs and returns the contents.
func ReadFile(fs Opener, path string) ([]byte, error) {
	rc, err := fs.Open(path)
	if err != nil {
		return nil, err
	}
	defer rc.Close()
	return io.ReadAll(rc)
}
