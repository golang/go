// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file defines types for abstract file system access and
// provides an implementation accessing the file system of the
// underlying OS.

package main

import (
	"fmt"
	"io"
	"io/ioutil"
	"os"
)

// The FileSystem interface specifies the methods godoc is using
// to access the file system for which it serves documentation.
type FileSystem interface {
	Open(path string) (io.ReadCloser, error)
	Lstat(path string) (os.FileInfo, error)
	Stat(path string) (os.FileInfo, error)
	ReadDir(path string) ([]os.FileInfo, error)
}

// ReadFile reads the file named by path from fs and returns the contents.
func ReadFile(fs FileSystem, path string) ([]byte, error) {
	rc, err := fs.Open(path)
	if err != nil {
		return nil, err
	}
	defer rc.Close()
	return ioutil.ReadAll(rc)
}

// ----------------------------------------------------------------------------
// OS-specific FileSystem implementation

var OS FileSystem = osFS{}

// osFS is the OS-specific implementation of FileSystem
type osFS struct{}

func (osFS) Open(path string) (io.ReadCloser, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	fi, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if fi.IsDir() {
		return nil, fmt.Errorf("Open: %s is a directory", path)
	}
	return f, nil
}

func (osFS) Lstat(path string) (os.FileInfo, error) {
	return os.Lstat(path)
}

func (osFS) Stat(path string) (os.FileInfo, error) {
	return os.Stat(path)
}

func (osFS) ReadDir(path string) ([]os.FileInfo, error) {
	return ioutil.ReadDir(path) // is sorted
}
