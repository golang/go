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
	"io"
	"net/http"
	"os"
	"path"
	"sort"
	"strings"
	"time"
)

type fileInfo struct {
	name  string
	mode  os.FileMode
	size  int64
	mtime time.Time
}

func (fi *fileInfo) Name() string       { return fi.name }
func (fi *fileInfo) Mode() os.FileMode  { return fi.mode }
func (fi *fileInfo) Size() int64        { return fi.size }
func (fi *fileInfo) ModTime() time.Time { return fi.mtime }
func (fi *fileInfo) IsDir() bool        { return fi.mode.IsDir() }
func (fi *fileInfo) Sys() interface{}   { return nil }

// httpZipFile is the zip-file based implementation of http.File
type httpZipFile struct {
	path          string // absolute path within zip FS without leading '/'
	info          os.FileInfo
	io.ReadCloser // nil for directory
	list          zipList
}

func (f *httpZipFile) Close() error {
	if !f.info.IsDir() {
		return f.ReadCloser.Close()
	}
	f.list = nil
	return nil
}

func (f *httpZipFile) Stat() (os.FileInfo, error) {
	return f.info, nil
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
		var mode os.FileMode
		var size int64
		var mtime time.Time
		if i := strings.IndexRune(name, '/'); i >= 0 {
			// We infer directories from files in subdirectories.
			// If we have x/y, return a directory entry for x.
			name = name[0:i] // keep local directory name only
			mode = os.ModeDir
			// no size or mtime for directories
		} else {
			mode = 0
			size = int64(e.UncompressedSize)
			mtime = e.ModTime()
		}
		// If we have x/y and x/z, don't return two directory entries for x.
		// TODO(gri): It should be possible to do this more efficiently
		// by determining the (fs.list) range of local directory entries
		// (via two binary searches).
		if name != prevname {
			list = append(list, &fileInfo{
				name,
				mode,
				size,
				mtime,
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
	return 0, fmt.Errorf("Seek not implemented for zip file entry: %s", f.info.Name())
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
			&fileInfo{
				name,
				0,
				int64(f.UncompressedSize),
				f.ModTime(),
			},
			rc,
			nil,
		}, nil
	}

	// not an exact match - must be a directory
	return &httpZipFile{
		path,
		&fileInfo{
			name,
			os.ModeDir,
			0,           // no size for directory
			time.Time{}, // no mtime for directory
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
