// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package vfs defines types for abstract file system access and provides an
// implementation accessing the file system of the underlying OS.
package vfs // import "golang.org/x/tools/godoc/vfs"

import (
	"io"
	"io/ioutil"
	"os"
)

// RootType indicates the type of files contained within a directory.
//
// The two main types are a GOROOT or a GOPATH directory.
type RootType string

const (
	RootTypeGoRoot     RootType = "GOROOT"
	RootTypeGoPath     RootType = "GOPATH"
	RootTypeStandAlone RootType = "StandAlone" // used by emptyvfs
	RootTypeAsset      RootType = "Assets"     // serves static assets using mapfs
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
	return ioutil.ReadAll(rc)
}
