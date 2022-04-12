// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package zipfs file provides an implementation of the FileSystem
// interface based on the contents of a .zip file.
//
// Assumptions:
//
//   - The file paths stored in the zip file must use a slash ('/') as path
//     separator; and they must be relative (i.e., they must not start with
//     a '/' - this is usually the case if the file was created w/o special
//     options).
//   - The zip file system treats the file paths found in the zip internally
//     like absolute paths w/o a leading '/'; i.e., the paths are considered
//     relative to the root of the file system.
//   - All path arguments to file system methods must be absolute paths.
package zipfs // import "golang.org/x/tools/godoc/vfs/zipfs"

import (
	"archive/zip"
	"fmt"
	"go/build"
	"io"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"golang.org/x/tools/godoc/vfs"
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
	if f := fi.file; f != nil {
		return int64(f.UncompressedSize)
	}
	return 0 // directory
}

func (fi zipFI) ModTime() time.Time {
	if f := fi.file; f != nil {
		return f.ModTime()
	}
	return time.Time{} // directory has no modified time entry
}

func (fi zipFI) Mode() os.FileMode {
	if fi.file == nil {
		// Unix directories typically are executable, hence 555.
		return os.ModeDir | 0555
	}
	return 0444
}

func (fi zipFI) IsDir() bool {
	return fi.file == nil
}

func (fi zipFI) Sys() interface{} {
	return nil
}

// zipFS is the zip-file based implementation of FileSystem
type zipFS struct {
	*zip.ReadCloser
	list zipList
	name string
}

func (fs *zipFS) String() string {
	return "zip(" + fs.name + ")"
}

func (fs *zipFS) RootType(abspath string) vfs.RootType {
	var t vfs.RootType
	switch {
	case exists(path.Join(vfs.GOROOT, abspath)):
		t = vfs.RootTypeGoRoot
	case isGoPath(abspath):
		t = vfs.RootTypeGoPath
	}
	return t
}

func isGoPath(abspath string) bool {
	for _, p := range filepath.SplitList(build.Default.GOPATH) {
		if exists(path.Join(p, abspath)) {
			return true
		}
	}
	return false
}

func exists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

func (fs *zipFS) Close() error {
	fs.list = nil
	return fs.ReadCloser.Close()
}

func zipPath(name string) (string, error) {
	name = path.Clean(name)
	if !path.IsAbs(name) {
		return "", fmt.Errorf("stat: not an absolute path: %s", name)
	}
	return name[1:], nil // strip leading '/'
}

func isRoot(abspath string) bool {
	return path.Clean(abspath) == "/"
}

func (fs *zipFS) stat(abspath string) (int, zipFI, error) {
	if isRoot(abspath) {
		return 0, zipFI{
			name: "",
			file: nil,
		}, nil
	}
	zippath, err := zipPath(abspath)
	if err != nil {
		return 0, zipFI{}, err
	}
	i, exact := fs.list.lookup(zippath)
	if i < 0 {
		// zippath has leading '/' stripped - print it explicitly
		return -1, zipFI{}, &os.PathError{Path: "/" + zippath, Err: os.ErrNotExist}
	}
	_, name := path.Split(zippath)
	var file *zip.File
	if exact {
		file = fs.list[i] // exact match found - must be a file
	}
	return i, zipFI{name, file}, nil
}

func (fs *zipFS) Open(abspath string) (vfs.ReadSeekCloser, error) {
	_, fi, err := fs.stat(abspath)
	if err != nil {
		return nil, err
	}
	if fi.IsDir() {
		return nil, fmt.Errorf("Open: %s is a directory", abspath)
	}
	r, err := fi.file.Open()
	if err != nil {
		return nil, err
	}
	return &zipSeek{fi.file, r}, nil
}

type zipSeek struct {
	file *zip.File
	io.ReadCloser
}

func (f *zipSeek) Seek(offset int64, whence int) (int64, error) {
	if whence == 0 && offset == 0 {
		r, err := f.file.Open()
		if err != nil {
			return 0, err
		}
		f.Close()
		f.ReadCloser = r
		return 0, nil
	}
	return 0, fmt.Errorf("unsupported Seek in %s", f.file.Name)
}

func (fs *zipFS) Lstat(abspath string) (os.FileInfo, error) {
	_, fi, err := fs.stat(abspath)
	return fi, err
}

func (fs *zipFS) Stat(abspath string) (os.FileInfo, error) {
	_, fi, err := fs.stat(abspath)
	return fi, err
}

func (fs *zipFS) ReadDir(abspath string) ([]os.FileInfo, error) {
	i, fi, err := fs.stat(abspath)
	if err != nil {
		return nil, err
	}
	if !fi.IsDir() {
		return nil, fmt.Errorf("ReadDir: %s is not a directory", abspath)
	}

	var list []os.FileInfo

	// make dirname the prefix that file names must start with to be considered
	// in this directory. we must special case the root directory because, per
	// the spec of this package, zip file entries MUST NOT start with /, so we
	// should not append /, as we would in every other case.
	var dirname string
	if isRoot(abspath) {
		dirname = ""
	} else {
		zippath, err := zipPath(abspath)
		if err != nil {
			return nil, err
		}
		dirname = zippath + "/"
	}
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

func New(rc *zip.ReadCloser, name string) vfs.FileSystem {
	list := make(zipList, len(rc.File))
	copy(list, rc.File) // sort a copy of rc.File
	sort.Sort(list)
	return &zipFS{rc, list, name}
}

type zipList []*zip.File

// zipList implements sort.Interface
func (z zipList) Len() int           { return len(z) }
func (z zipList) Less(i, j int) bool { return z[i].Name < z[j].Name }
func (z zipList) Swap(i, j int)      { z[i], z[j] = z[j], z[i] }

// lookup returns the smallest index of an entry with an exact match
// for name, or an inexact match starting with name/. If there is no
// such entry, the result is -1, false.
func (z zipList) lookup(name string) (index int, exact bool) {
	// look for exact match first (name comes before name/ in z)
	i := sort.Search(len(z), func(i int) bool {
		return name <= z[i].Name
	})
	if i >= len(z) {
		return -1, false
	}
	// 0 <= i < len(z)
	if z[i].Name == name {
		return i, true
	}

	// look for inexact match (must be in z[i:], if present)
	z = z[i:]
	name += "/"
	j := sort.Search(len(z), func(i int) bool {
		return name <= z[i].Name
	})
	if j >= len(z) {
		return -1, false
	}
	// 0 <= j < len(z)
	if strings.HasPrefix(z[j].Name, name) {
		return i + j, false
	}

	return -1, false
}
