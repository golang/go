// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vfs

import (
	"fmt"
	"io/ioutil"
	"log"
	"os"
	pathpkg "path"
	"path/filepath"
	"strings"
)

// OS returns an implementation of FileSystem reading from the
// tree rooted at root.  Recording a root is convenient everywhere
// but necessary on Windows, because the slash-separated path
// passed to Open has no way to specify a drive letter.  Using a root
// lets code refer to OS(`c:\`), OS(`d:\`) and so on.
func OS(root string) FileSystem {
	return osFS(root)
}

type osFS string

func (root osFS) String() string { return "os(" + string(root) + ")" }

func (root osFS) resolve(path string) string {
	// Clean the path so that it cannot possibly begin with ../.
	// If it did, the result of filepath.Join would be outside the
	// tree rooted at root.  We probably won't ever see a path
	// with .. in it, but be safe anyway.
	path = pathpkg.Clean("/" + path)

	return filepath.Join(string(root), path)
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
	return stat(root.resolve(path))
}

var readdir = ioutil.ReadDir // for testing
var stat = os.Stat           // for testing

func (root osFS) ReadDir(path string) ([]os.FileInfo, error) {
	fis, err := readdir(root.resolve(path))
	if err != nil {
		return fis, err
	}
	ret := fis[:0]

	// reread the files with os.Stat since they might be symbolic links
	for _, fi := range fis {
		if fi.Mode()&os.ModeSymlink != 0 {
			baseName := fi.Name()
			fi, err = root.Stat(pathpkg.Join(path, baseName))
			if err != nil {
				if os.IsNotExist(err) && strings.HasPrefix(baseName, ".") {
					// Ignore editor spam files without log spam.
					continue
				}
				log.Printf("ignoring symlink: %v", err)
				continue
			}
		}
		ret = append(ret, fi)
	}

	return ret, nil // is sorted
}
