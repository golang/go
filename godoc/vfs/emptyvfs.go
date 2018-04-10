// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vfs

import (
	"fmt"
	"os"
	"time"
)

// NewNameSpace returns a NameSpace pre-initialized with an empty
// emulated directory mounted on the root mount point "/". This
// allows directory traversal routines to work properly even if
// a folder is not explicitly mounted at root by the user.
func NewNameSpace() NameSpace {
	ns := NameSpace{}
	ns.Bind("/", &emptyVFS{}, "/", BindReplace)
	return ns
}

// type emptyVFS emulates a FileSystem consisting of an empty directory
type emptyVFS struct{}

// Open implements Opener. Since emptyVFS is an empty directory, all
// attempts to open a file should returns errors.
func (e *emptyVFS) Open(path string) (ReadSeekCloser, error) {
	if path == "/" {
		return nil, fmt.Errorf("open: / is a directory")
	}
	return nil, os.ErrNotExist
}

// Stat returns os.FileInfo  for an empty directory if the path is
// is root "/" or error. os.FileInfo is implemented by emptyVFS
func (e *emptyVFS) Stat(path string) (os.FileInfo, error) {
	if path == "/" {
		return e, nil
	}
	return nil, os.ErrNotExist
}

func (e *emptyVFS) Lstat(path string) (os.FileInfo, error) {
	return e.Stat(path)
}

// ReadDir returns an empty os.FileInfo slice for "/", else error.
func (e *emptyVFS) ReadDir(path string) ([]os.FileInfo, error) {
	if path == "/" {
		return []os.FileInfo{}, nil
	}
	return nil, os.ErrNotExist
}

func (e *emptyVFS) String() string {
	return "emptyVFS(/)"
}

func (e *emptyVFS) RootType(path string) RootType {
	return ""
}

// These functions below implement os.FileInfo for the single
// empty emulated directory.

func (e *emptyVFS) Name() string {
	return "/"
}

func (e *emptyVFS) Size() int64 {
	return 0
}

func (e *emptyVFS) Mode() os.FileMode {
	return os.ModeDir | os.ModePerm
}

func (e *emptyVFS) ModTime() time.Time {
	return time.Time{}
}

func (e *emptyVFS) IsDir() bool {
	return true
}

func (e *emptyVFS) Sys() interface{} {
	return nil
}
