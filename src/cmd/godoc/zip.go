// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file provides an implementation of the FileSystem
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
// - All path arguments to file system methods must be absolute paths.

package main

import (
	"archive/zip"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path"
	"sort"
	"strings"
)

// zipFI is the zip-file based implementation of FileInfo
type zipFI struct {
	name string    // directory-local name
	file *zip.File // nil for a directory
}

func (fi zipFI) Name() string {
	return fi.name
}

func (fi zipFI) Size() int64 {
	if fi.file != nil {
		return int64(fi.file.UncompressedSize)
	}
	return 0 // directory
}

func (fi zipFI) IsDirectory() bool {
	return fi.file == nil
}

func (fi zipFI) IsRegular() bool {
	return fi.file != nil
}

// zipFS is the zip-file based implementation of FileSystem
type zipFS struct {
	*zip.ReadCloser
	list zipList
}

func (fs *zipFS) Close() os.Error {
	fs.list = nil
	return fs.ReadCloser.Close()
}

func zipPath(name string) string {
	if !path.IsAbs(name) {
		panic(fmt.Sprintf("stat: not an absolute path: %s", name))
	}
	return name[1:] // strip '/'
}

func (fs *zipFS) stat(abspath string) (int, zipFI, os.Error) {
	i := fs.list.lookup(abspath)
	if i < 0 {
		return -1, zipFI{}, fmt.Errorf("file not found: %s", abspath)
	}
	var file *zip.File
	if abspath == fs.list[i].Name {
		file = fs.list[i] // exact match found - must be a file
	}
	_, name := path.Split(abspath)
	return i, zipFI{name, file}, nil
}

func (fs *zipFS) Open(abspath string) (io.ReadCloser, os.Error) {
	_, fi, err := fs.stat(zipPath(abspath))
	if err != nil {
		return nil, err
	}
	if fi.IsDirectory() {
		return nil, fmt.Errorf("Open: %s is a directory", abspath)
	}
	return fi.file.Open()
}

func (fs *zipFS) Lstat(abspath string) (FileInfo, os.Error) {
	_, fi, err := fs.stat(zipPath(abspath))
	return fi, err
}

func (fs *zipFS) Stat(abspath string) (FileInfo, os.Error) {
	_, fi, err := fs.stat(zipPath(abspath))
	return fi, err
}

func (fs *zipFS) ReadDir(abspath string) ([]FileInfo, os.Error) {
	path := zipPath(abspath)
	i, fi, err := fs.stat(path)
	if err != nil {
		return nil, err
	}
	if !fi.IsDirectory() {
		return nil, fmt.Errorf("ReadDir: %s is not a directory", abspath)
	}

	var list []FileInfo
	dirname := path + "/"
	prevname := ""
	for _, e := range fs.list[i:] {
		if !strings.HasPrefix(e.Name, dirname) {
			break // not in the same directory anymore
		}
		name := e.Name[len(dirname):] // local name
		file := e
		if i := strings.IndexRune(name, '/'); i >= 0 {
			// We infer directories from files in subdirectories.
			// If we have x/y, return a directory entry for x.
			name = name[0:i] // keep local directory name only
			file = nil
		}
		// If we have x/y and x/z, don't return two directory entries for x.
		// TODO(gri): It should be possible to do this more efficiently
		// by determining the (fs.list) range of local directory entries
		// (via two binary searches).
		if name != prevname {
			list = append(list, zipFI{name, file})
			prevname = name
		}
	}

	return list, nil
}

func (fs *zipFS) ReadFile(abspath string) ([]byte, os.Error) {
	rc, err := fs.Open(abspath)
	if err != nil {
		return nil, err
	}
	return ioutil.ReadAll(rc)
}

func NewZipFS(rc *zip.ReadCloser) FileSystem {
	list := make(zipList, len(rc.File))
	copy(list, rc.File) // sort a copy of rc.File
	sort.Sort(list)
	return &zipFS{rc, list}
}

type zipList []*zip.File

// zipList implements sort.Interface
func (z zipList) Len() int           { return len(z) }
func (z zipList) Less(i, j int) bool { return z[i].Name < z[j].Name }
func (z zipList) Swap(i, j int)      { z[i], z[j] = z[j], z[i] }

// lookup returns the first index in the zipList
// of a path equal to name or beginning with name/.
func (z zipList) lookup(name string) int {
	i := sort.Search(len(z), func(i int) bool {
		return name <= z[i].Name
	})
	if i >= 0 {
		iname := z[i].Name
		if strings.HasPrefix(iname, name) && (len(name) == len(iname) || iname[len(name)] == '/') {
			return i
		}
	}
	return -1 // no match
}
