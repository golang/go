// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.16
// +build go1.16

package vfs

import (
	"io/fs"
	"os"
	"path"
	"strings"
)

// FromFS converts an fs.FS to the FileSystem interface.
func FromFS(fsys fs.FS) FileSystem {
	return &fsysToFileSystem{fsys}
}

type fsysToFileSystem struct {
	fsys fs.FS
}

func (f *fsysToFileSystem) fsPath(name string) string {
	name = path.Clean(name)
	if name == "/" {
		return "."
	}
	return strings.TrimPrefix(name, "/")
}

func (f *fsysToFileSystem) Open(name string) (ReadSeekCloser, error) {
	file, err := f.fsys.Open(f.fsPath(name))
	if err != nil {
		return nil, err
	}
	if rsc, ok := file.(ReadSeekCloser); ok {
		return rsc, nil
	}
	return &noSeekFile{f.fsPath(name), file}, nil
}

func (f *fsysToFileSystem) Lstat(name string) (os.FileInfo, error) {
	return fs.Stat(f.fsys, f.fsPath(name))
}

func (f *fsysToFileSystem) Stat(name string) (os.FileInfo, error) {
	return fs.Stat(f.fsys, f.fsPath(name))
}

func (f *fsysToFileSystem) RootType(name string) RootType { return "" }

func (f *fsysToFileSystem) ReadDir(name string) ([]os.FileInfo, error) {
	dirs, err := fs.ReadDir(f.fsys, f.fsPath(name))
	var infos []os.FileInfo
	for _, d := range dirs {
		info, err1 := d.Info()
		if err1 != nil {
			if err == nil {
				err = err1
			}
			continue
		}
		infos = append(infos, info)
	}
	return infos, err
}

func (f *fsysToFileSystem) String() string { return "io/fs" }

type noSeekFile struct {
	path string
	fs.File
}

func (f *noSeekFile) Seek(offset int64, whence int) (int64, error) {
	return 0, &fs.PathError{Op: "seek", Path: f.path, Err: fs.ErrInvalid}
}
