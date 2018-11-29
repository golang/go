// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modfetch

import (
	"archive/zip"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path"
	"path/filepath"
	"strings"

	"cmd/go/internal/modfetch/codehost"
	"cmd/go/internal/module"
	"cmd/go/internal/str"
)

func Unzip(dir, zipfile, prefix string, maxSize int64) error {
	// TODO(bcmills): The maxSize parameter is invariantly 0. Remove it.
	if maxSize == 0 {
		maxSize = codehost.MaxZipFile
	}

	// Directory can exist, but must be empty.
	files, _ := ioutil.ReadDir(dir)
	if len(files) > 0 {
		return fmt.Errorf("target directory %v exists and is not empty", dir)
	}
	if err := os.MkdirAll(dir, 0777); err != nil {
		return err
	}

	f, err := os.Open(zipfile)
	if err != nil {
		return err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return err
	}

	z, err := zip.NewReader(f, info.Size())
	if err != nil {
		return fmt.Errorf("unzip %v: %s", zipfile, err)
	}

	foldPath := make(map[string]string)
	var checkFold func(string) error
	checkFold = func(name string) error {
		fold := str.ToFold(name)
		if foldPath[fold] == name {
			return nil
		}
		dir := path.Dir(name)
		if dir != "." {
			if err := checkFold(dir); err != nil {
				return err
			}
		}
		if foldPath[fold] == "" {
			foldPath[fold] = name
			return nil
		}
		other := foldPath[fold]
		return fmt.Errorf("unzip %v: case-insensitive file name collision: %q and %q", zipfile, other, name)
	}

	// Check total size, valid file names.
	var size int64
	for _, zf := range z.File {
		if !str.HasPathPrefix(zf.Name, prefix) {
			return fmt.Errorf("unzip %v: unexpected file name %s", zipfile, zf.Name)
		}
		if zf.Name == prefix || strings.HasSuffix(zf.Name, "/") {
			continue
		}
		name := zf.Name[len(prefix)+1:]
		if err := module.CheckFilePath(name); err != nil {
			return fmt.Errorf("unzip %v: %v", zipfile, err)
		}
		if err := checkFold(name); err != nil {
			return err
		}
		if path.Clean(zf.Name) != zf.Name || strings.HasPrefix(zf.Name[len(prefix)+1:], "/") {
			return fmt.Errorf("unzip %v: invalid file name %s", zipfile, zf.Name)
		}
		s := int64(zf.UncompressedSize64)
		if s < 0 || maxSize-size < s {
			return fmt.Errorf("unzip %v: content too large", zipfile)
		}
		size += s
	}

	// Unzip, enforcing sizes checked earlier.
	for _, zf := range z.File {
		if zf.Name == prefix || strings.HasSuffix(zf.Name, "/") {
			continue
		}
		name := zf.Name[len(prefix):]
		dst := filepath.Join(dir, name)
		if err := os.MkdirAll(filepath.Dir(dst), 0777); err != nil {
			return err
		}
		w, err := os.OpenFile(dst, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0444)
		if err != nil {
			return fmt.Errorf("unzip %v: %v", zipfile, err)
		}
		r, err := zf.Open()
		if err != nil {
			w.Close()
			return fmt.Errorf("unzip %v: %v", zipfile, err)
		}
		lr := &io.LimitedReader{R: r, N: int64(zf.UncompressedSize64) + 1}
		_, err = io.Copy(w, lr)
		r.Close()
		if err != nil {
			w.Close()
			return fmt.Errorf("unzip %v: %v", zipfile, err)
		}
		if err := w.Close(); err != nil {
			return fmt.Errorf("unzip %v: %v", zipfile, err)
		}
		if lr.N <= 0 {
			return fmt.Errorf("unzip %v: content too large", zipfile)
		}
	}

	return nil
}

// makeDirsReadOnly makes a best-effort attempt to remove write permissions for dir
// and its transitive contents.
func makeDirsReadOnly(dir string) {
	type pathMode struct {
		path string
		mode os.FileMode
	}
	var dirs []pathMode // in lexical order
	filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err == nil && info.Mode()&0222 != 0 {
			if info.IsDir() {
				dirs = append(dirs, pathMode{path, info.Mode()})
			}
		}
		return nil
	})

	// Run over list backward to chmod children before parents.
	for i := len(dirs) - 1; i >= 0; i-- {
		os.Chmod(dirs[i].path, dirs[i].mode&^0222)
	}
}

// RemoveAll removes a directory written by Download or Unzip, first applying
// any permission changes needed to do so.
func RemoveAll(dir string) error {
	// Module cache has 0555 directories; make them writable in order to remove content.
	filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil // ignore errors walking in file system
		}
		if info.IsDir() {
			os.Chmod(path, 0777)
		}
		return nil
	})
	return os.RemoveAll(dir)
}
