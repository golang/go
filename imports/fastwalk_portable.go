// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !linux,!darwin,!freebsd,!openbsd,!netbsd

package imports

import (
	"io/ioutil"
	"os"
)

// readDir calls fn for each directory entry in dirName.
// It does not descend into directories or follow symlinks.
// If fn returns a non-nil error, readDir returns with that error
// immediately.
func readDir(dirName string, fn func(dirName, entName string, typ os.FileMode) error) error {
	fis, err := ioutil.ReadDir(dirName)
	if err != nil {
		return err
	}
	for _, fi := range fis {
		if err := fn(dirName, fi.Name(), fi.Mode()&os.ModeType); err != nil {
			return err
		}
	}
	return nil
}
