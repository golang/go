// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vfs

import (
	"fmt"
	"go/build"
	"io/ioutil"
	"os"
	pathpkg "path"
	"path/filepath"
	"runtime"
)

// We expose a new variable because otherwise we need to copy the findGOROOT logic again
// from cmd/godoc which is already copied twice from the standard library.

// GOROOT returns the GOROOT path under which the godoc binary is running.
// It is needed to check whether a filesystem root is under GOROOT or not.
// This is set from cmd/godoc/main.go
var GOROOT = runtime.GOROOT()

// OS returns an implementation of FileSystem reading from the
// tree rooted at root.  Recording a root is convenient everywhere
// but necessary on Windows, because the slash-separated path
// passed to Open has no way to specify a drive letter.  Using a root
// lets code refer to OS(`c:\`), OS(`d:\`) and so on.
func OS(root string) FileSystem {
	var t RootType
	switch {
	case root == GOROOT:
		t = RootTypeGoRoot
	case isGoPath(root):
		t = RootTypeGoPath
	}
	return osFS{rootPath: root, rootType: t}
}

type osFS struct {
	rootPath string
	rootType RootType
}

func isGoPath(path string) bool {
	for _, bp := range filepath.SplitList(build.Default.GOPATH) {
		for _, gp := range filepath.SplitList(path) {
			if bp == gp {
				return true
			}
		}
	}
	return false
}

func (root osFS) String() string { return "os(" + root.rootPath + ")" }

// RootType returns the root type for the filesystem.
//
// Note that we ignore the path argument because roottype is a property of
// this filesystem. But for other filesystems, the roottype might need to be
// dynamically deduced at call time.
func (root osFS) RootType(path string) RootType {
	return root.rootType
}

func (root osFS) resolve(path string) string {
	// Clean the path so that it cannot possibly begin with ../.
	// If it did, the result of filepath.Join would be outside the
	// tree rooted at root.  We probably won't ever see a path
	// with .. in it, but be safe anyway.
	path = pathpkg.Clean("/" + path)

	return filepath.Join(root.rootPath, path)
}

func (root osFS) Open(path string) (ReadSeekCloser, error) {
	f, err := os.Open(root.resolve(path))
	if err != nil {
		return nil, err
	}
	fi, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, err
	}
	if fi.IsDir() {
		f.Close()
		return nil, fmt.Errorf("Open: %s is a directory", path)
	}
	return f, nil
}

func (root osFS) Lstat(path string) (os.FileInfo, error) {
	return os.Lstat(root.resolve(path))
}

func (root osFS) Stat(path string) (os.FileInfo, error) {
	return os.Stat(root.resolve(path))
}

func (root osFS) ReadDir(path string) ([]os.FileInfo, error) {
	return ioutil.ReadDir(root.resolve(path)) // is sorted
}
