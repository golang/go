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


// The FileInfo interface provides access to file information.
type FileInfo interface {
	Name() string
	Size() int64
	IsRegular() bool
	IsDirectory() bool
}


// The FileSystem interface specifies the methods godoc is using
// to access the file system for which it serves documentation.
type FileSystem interface {
	Open(path string) (io.ReadCloser, os.Error)
	Lstat(path string) (FileInfo, os.Error)
	Stat(path string) (FileInfo, os.Error)
	ReadDir(path string) ([]FileInfo, os.Error)
	ReadFile(path string) ([]byte, os.Error)
}


// ----------------------------------------------------------------------------
// OS-specific FileSystem implementation

var OS FileSystem = osFS{}


// osFI is the OS-specific implementation of FileInfo.
type osFI struct {
	*os.FileInfo
}


func (fi osFI) Name() string {
	return fi.FileInfo.Name
}


func (fi osFI) Size() int64 {
	if fi.IsDirectory() {
		return 0
	}
	return fi.FileInfo.Size
}


// osFS is the OS-specific implementation of FileSystem
type osFS struct{}

func (osFS) Open(path string) (io.ReadCloser, os.Error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	fi, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if fi.IsDirectory() {
		return nil, fmt.Errorf("Open: %s is a directory", path)
	}
	return f, nil
}


func (osFS) Lstat(path string) (FileInfo, os.Error) {
	fi, err := os.Lstat(path)
	return osFI{fi}, err
}


func (osFS) Stat(path string) (FileInfo, os.Error) {
	fi, err := os.Stat(path)
	return osFI{fi}, err
}


func (osFS) ReadDir(path string) ([]FileInfo, os.Error) {
	l0, err := ioutil.ReadDir(path) // l0 is sorted
	if err != nil {
		return nil, err
	}
	l1 := make([]FileInfo, len(l0))
	for i, e := range l0 {
		l1[i] = osFI{e}
	}
	return l1, nil
}


func (osFS) ReadFile(path string) ([]byte, os.Error) {
	return ioutil.ReadFile(path)
}
