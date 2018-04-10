// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package mapfs file provides an implementation of the FileSystem
// interface based on the contents of a map[string]string.
package mapfs // import "golang.org/x/tools/godoc/vfs/mapfs"

import (
	"io"
	"os"
	pathpkg "path"
	"sort"
	"strings"
	"time"

	"golang.org/x/tools/godoc/vfs"
)

// New returns a new FileSystem from the provided map.
// Map keys should be forward slash-separated pathnames
// and not contain a leading slash.
func New(m map[string]string) vfs.FileSystem {
	return mapFS(m)
}

// mapFS is the map based implementation of FileSystem
type mapFS map[string]string

func (fs mapFS) String() string { return "mapfs" }

func (fs mapFS) RootType(p string) vfs.RootType {
	return ""
}

func (fs mapFS) Close() error { return nil }

func filename(p string) string {
	return strings.TrimPrefix(p, "/")
}

func (fs mapFS) Open(p string) (vfs.ReadSeekCloser, error) {
	b, ok := fs[filename(p)]
	if !ok {
		return nil, os.ErrNotExist
	}
	return nopCloser{strings.NewReader(b)}, nil
}

func fileInfo(name, contents string) os.FileInfo {
	return mapFI{name: pathpkg.Base(name), size: len(contents)}
}

func dirInfo(name string) os.FileInfo {
	return mapFI{name: pathpkg.Base(name), dir: true}
}

func (fs mapFS) Lstat(p string) (os.FileInfo, error) {
	b, ok := fs[filename(p)]
	if ok {
		return fileInfo(p, b), nil
	}
	ents, _ := fs.ReadDir(p)
	if len(ents) > 0 {
		return dirInfo(p), nil
	}
	return nil, os.ErrNotExist
}

func (fs mapFS) Stat(p string) (os.FileInfo, error) {
	return fs.Lstat(p)
}

// slashdir returns path.Dir(p), but special-cases paths not beginning
// with a slash to be in the root.
func slashdir(p string) string {
	d := pathpkg.Dir(p)
	if d == "." {
		return "/"
	}
	if strings.HasPrefix(p, "/") {
		return d
	}
	return "/" + d
}

func (fs mapFS) ReadDir(p string) ([]os.FileInfo, error) {
	p = pathpkg.Clean(p)
	var ents []string
	fim := make(map[string]os.FileInfo) // base -> fi
	for fn, b := range fs {
		dir := slashdir(fn)
		isFile := true
		var lastBase string
		for {
			if dir == p {
				base := lastBase
				if isFile {
					base = pathpkg.Base(fn)
				}
				if fim[base] == nil {
					var fi os.FileInfo
					if isFile {
						fi = fileInfo(fn, b)
					} else {
						fi = dirInfo(base)
					}
					ents = append(ents, base)
					fim[base] = fi
				}
			}
			if dir == "/" {
				break
			} else {
				isFile = false
				lastBase = pathpkg.Base(dir)
				dir = pathpkg.Dir(dir)
			}
		}
	}
	if len(ents) == 0 {
		return nil, os.ErrNotExist
	}

	sort.Strings(ents)
	var list []os.FileInfo
	for _, dir := range ents {
		list = append(list, fim[dir])
	}
	return list, nil
}

// mapFI is the map-based implementation of FileInfo.
type mapFI struct {
	name string
	size int
	dir  bool
}

func (fi mapFI) IsDir() bool        { return fi.dir }
func (fi mapFI) ModTime() time.Time { return time.Time{} }
func (fi mapFI) Mode() os.FileMode {
	if fi.IsDir() {
		return 0755 | os.ModeDir
	}
	return 0444
}
func (fi mapFI) Name() string     { return pathpkg.Base(fi.name) }
func (fi mapFI) Size() int64      { return int64(fi.size) }
func (fi mapFI) Sys() interface{} { return nil }

type nopCloser struct {
	io.ReadSeeker
}

func (nc nopCloser) Close() error { return nil }
