// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package ioutil implements some I/O utility functions.
//
// Deprecated: As of Go 1.16, the same functionality is now provided
// by package io or package os, and those implementations
// should be preferred in new code.
// See the specific function documentation for details.
package ioutil

import (
	"io"
	"io/fs"
	"os"
	"sort"
)

// ReadAll reads from r until an error or EOF and returns the data it read.
// A successful call returns err == nil, not err == EOF. Because ReadAll is
// defined to read from src until EOF, it does not treat an EOF from Read
// as an error to be reported.
//
// Deprecated: As of Go 1.16, this function simply calls io.ReadAll.
func ReadAll(r io.Reader) ([]byte, error) {
	return io.ReadAll(r)
}

// ReadFile reads the file named by filename and returns the contents.
// A successful call returns err == nil, not err == EOF. Because ReadFile
// reads the whole file, it does not treat an EOF from Read as an error
// to be reported.
//
// Deprecated: As of Go 1.16, this function simply calls os.ReadFile.
func ReadFile(filename string) ([]byte, error) {
	return os.ReadFile(filename)
}

// WriteFile writes data to a file named by filename.
// If the file does not exist, WriteFile creates it with permissions perm
// (before umask); otherwise WriteFile truncates it before writing, without changing permissions.
//
// Deprecated: As of Go 1.16, this function simply calls os.WriteFile.
func WriteFile(filename string, data []byte, perm fs.FileMode) error {
	return os.WriteFile(filename, data, perm)
}

// ReadDir reads the directory named by dirname and returns
// a list of fs.FileInfo for the directory's contents,
// sorted by filename. If an error occurs reading the directory,
// ReadDir returns no directory entries along with the error.
//
// Deprecated: As of Go 1.16, os.ReadDir is a more efficient and correct choice:
// it returns a list of fs.DirEntry instead of fs.FileInfo,
// and it returns partial results in the case of an error
// midway through reading a directory.
//
// If you must continue obtaining a list of fs.FileInfo, you still can:
//
//	entries, err := os.ReadDir(dirname)
//	if err != nil { ... }
//	infos := make([]fs.FileInfo, 0, len(entries))
//	for _, entry := range entries {
//		info, err := entry.Info()
//		if err != nil { ... }
//		infos = append(infos, info)
//	}
func ReadDir(dirname string) ([]fs.FileInfo, error) {
	f, err := os.Open(dirname)
	if err != nil {
		return nil, err
	}
	list, err := f.Readdir(-1)
	f.Close()
	if err != nil {
		return nil, err
	}
	sort.Slice(list, func(i, j int) bool { return list[i].Name() < list[j].Name() })
	return list, nil
}

// NopCloser returns a ReadCloser with a no-op Close method wrapping
// the provided Reader r.
//
// Deprecated: As of Go 1.16, this function simply calls io.NopCloser.
func NopCloser(r io.Reader) io.ReadCloser {
	return io.NopCloser(r)
}

// Discard is an io.Writer on which all Write calls succeed
// without doing anything.
//
// Deprecated: As of Go 1.16, this value is simply io.Discard.
var Discard io.Writer = io.Discard
