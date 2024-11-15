// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fsys

import (
	"io/fs"
	"path/filepath"
)

// Walk walks the file tree rooted at root, calling walkFn for each file or
// directory in the tree, including root.
func Walk(root string, walkFn filepath.WalkFunc) error {
	Trace("Walk", root)
	info, err := Lstat(root)
	if err != nil {
		err = walkFn(root, nil, err)
	} else {
		err = walk(root, info, walkFn)
	}
	if err == filepath.SkipDir {
		return nil
	}
	return err
}

// walk recursively descends path, calling walkFn. Copied, with some
// modifications from path/filepath.walk.
func walk(path string, info fs.FileInfo, walkFn filepath.WalkFunc) error {
	if err := walkFn(path, info, nil); err != nil || !info.IsDir() {
		return err
	}

	fis, err := ReadDir(path)
	if err != nil {
		return walkFn(path, info, err)
	}

	for _, fi := range fis {
		filename := filepath.Join(path, fi.Name())
		if err := walk(filename, fi, walkFn); err != nil {
			if !fi.IsDir() || err != filepath.SkipDir {
				return err
			}
		}
	}
	return nil
}
