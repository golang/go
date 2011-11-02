// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file provides an implementation of the http.FileSystem
// interface based on the contents of a .zip file.
//
// Assumptions:
//
// - The file paths stored in the zip file must use a slash ('/') as path
//   separator; and they must be relative (i.e., they must not start with
//   a '/' - this is usually the case if the file was created w/o special
//   options).
// - The zip file system treats the file paths found in the zip internally
//   like absolute paths w/o a leading '/'; i.e., the paths are considered
//   relative to the root of the file system.
// - All path arguments to file system methods are considered relative to
//   the root specified with NewHttpZipFS (even if the paths start with a '/').

// TODO(gri) Should define a commonly used FileSystem API that is the same
//           for http and godoc. Then we only need one zip-file based file
//           system implementation.

package main

import (
	"archive/zip"
	"fmt"
	"http"
	"io"
	"os"
	"path"
	"sort"
	"strings"
)

// We cannot import syscall on app engine.
// TODO(gri) Once we have a truly abstract FileInfo implementation
//           this won't be needed anymore.
const (
	S_IFDIR = 0x4000 // == syscall.S_IFDIR
	S_IFREG = 0x8000 // == syscall.S_IFREG
)

// httpZipFile is the zip-file based implementation of http.File
type httpZipFile struct {
	path          string // absolute path within zip FS without leading '/'
	info          os.FileInfo
	io.ReadCloser // nil for directory
	list          zipList
}

func (f *httpZipFile) Close() error {
	if f.info.IsRegular() {
		return f.ReadCloser.Close()
	}
	f.list = nil
	return nil
}

func (f *httpZipFile) Stat() (*os.FileInfo, error) {
	return &f.info, nil
}

func (f *httpZipFile) Readdir(count int) ([]os.FileInfo, error) {
	var list []os.FileInfo
	dirname := f.path + "/"
	prevname := ""
	for i, e := range f.list {
		if count == 0 {
			f.list = f.list[i:]
			break
		}
		if !strings.HasPrefix(e.Name, dirname) {
			f.list = nil
			break // not in the same directory anymore
		}
		name := e.Name[len(dirname):] // local name
		var mode uint32
		var size, mtime_ns int64
		if i := strings.IndexRune(name, '/'); i >= 0 {
			// We infer directories from files in subdirectories.
			// If we have x/y, return a directory entry for x.
			name = name[0:i] // keep local directory name only
			mode = S_IFDIR
			// no size or mtime_ns for directories
		} else {
			mode = S_IFREG
			size = int64(e.UncompressedSize)
			mtime_ns = e.Mtime_ns()
		}
		// If we have x/y and x/z, don't return two directory entries for x.
		// TODO(gri): It should be possible to do this more efficiently
		// by determining the (fs.list) range of local directory entries
		// (via two binary searches).
		if name != prevname {
			list = append(list, os.FileInfo{
				Name:     name,
				Mode:     mode,
				Size:     size,
				Mtime_ns: mtime_ns,
			})
			prevname = name
			count--
		}
	}

	if count >= 0 && len(list) == 0 {
		return nil, io.EOF
	}

	return list, nil
}

func (f *httpZipFile) Seek(offset int64, whence int) (int64, error) {
	return 0, fmt.Errorf("Seek not implemented for zip file entry: %s", f.info.Name)
}

// httpZipFS is the zip-file based implementation of http.FileSystem
type httpZipFS struct {
	*zip.ReadCloser
	list zipList
	root string
}

func (fs *httpZipFS) Open(name string) (http.File, error) {
	// fs.root does not start with '/'.
	path := path.Join(fs.root, name) // path is clean
	index, exact := fs.list.lookup(path)
	if index < 0 || !strings.HasPrefix(path, fs.root) {
		// file not found or not under root
		return nil, fmt.Errorf("file not found: %s", name)
	}

	if exact {
		// exact match found - must be a file
		f := fs.list[index]
		rc, err := f.Open()
		if err != nil {
			return nil, err
		}
		return &httpZipFile{
			path,
			os.FileInfo{
				Name:     name,
				Mode:     S_IFREG,
				Size:     int64(f.UncompressedSize),
				Mtime_ns: f.Mtime_ns(),
			},
			rc,
			nil,
		}, nil
	}

	// not an exact match - must be a directory
	return &httpZipFile{
		path,
		os.FileInfo{
			Name: name,
			Mode: S_IFDIR,
			// no size or mtime_ns for directories
		},
		nil,
		fs.list[index:],
	}, nil
}

func (fs *httpZipFS) Close() error {
	fs.list = nil
	return fs.ReadCloser.Close()
}

// NewHttpZipFS creates a new http.FileSystem based on the contents of
// the zip file rc restricted to the directory tree specified by root;
// root must be an absolute path.
func NewHttpZipFS(rc *zip.ReadCloser, root string) http.FileSystem {
	list := make(zipList, len(rc.File))
	copy(list, rc.File) // sort a copy of rc.File
	sort.Sort(list)
	return &httpZipFS{rc, list, zipPath(root)}
}
