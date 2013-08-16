// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package mapfs file provides an implementation of the FileSystem
// interface based on the contents of a map[string]string.
package mapfs

import (
	"io"
	"os"
	"strings"
	"time"

	"code.google.com/p/go.tools/godoc/vfs"
)

func New(m map[string]string) vfs.FileSystem {
	return mapFS(m)
}

// mapFS is the map based implementation of FileSystem
type mapFS map[string]string

func (fs mapFS) String() string { return "mapfs" }

func (fs mapFS) Close() error { return nil }

func filename(p string) string {
	if len(p) > 0 && p[0] == '/' {
		p = p[1:]
	}
	return p
}

func (fs mapFS) Open(p string) (vfs.ReadSeekCloser, error) {
	b, ok := fs[filename(p)]
	if !ok {
		return nil, os.ErrNotExist
	}
	return nopCloser{strings.NewReader(b)}, nil
}

func (fs mapFS) Lstat(p string) (os.FileInfo, error) {
	b, ok := fs[filename(p)]
	if !ok {
		return nil, os.ErrNotExist
	}
	return mapFI{name: p, size: int64(len(b))}, nil
}

func (fs mapFS) Stat(p string) (os.FileInfo, error) {
	return fs.Lstat(p)
}

func (fs mapFS) ReadDir(p string) ([]os.FileInfo, error) {
	var list []os.FileInfo
	for fn, b := range fs {
		list = append(list, mapFI{name: fn, size: int64(len(b))})
	}
	return list, nil
}

// mapFI is the map-based implementation of FileInfo.
type mapFI struct {
	name string
	size int64
}

func (fi mapFI) IsDir() bool        { return false }
func (fi mapFI) ModTime() time.Time { return time.Time{} }
func (fi mapFI) Mode() os.FileMode  { return 0444 }
func (fi mapFI) Name() string       { return fi.name }
func (fi mapFI) Size() int64        { return fi.size }
func (fi mapFI) Sys() interface{}   { return nil }

type nopCloser struct {
	io.ReadSeeker
}

func (nc nopCloser) Close() error { return nil }
