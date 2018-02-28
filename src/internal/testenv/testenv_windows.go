// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testenv

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"sync"
	"syscall"
)

var symlinkOnce sync.Once
var winSymlinkErr error

func initWinHasSymlink() {
	tmpdir, err := ioutil.TempDir("", "symtest")
	if err != nil {
		panic("failed to create temp directory: " + err.Error())
	}
	defer os.RemoveAll(tmpdir)

	err = os.Symlink("target", filepath.Join(tmpdir, "symlink"))
	if err != nil {
		err = err.(*os.LinkError).Err
		switch err {
		case syscall.EWINDOWS, syscall.ERROR_PRIVILEGE_NOT_HELD:
			winSymlinkErr = err
		}
	}
}

func hasSymlink() (ok bool, reason string) {
	symlinkOnce.Do(initWinHasSymlink)

	switch winSymlinkErr {
	case nil:
		return true, ""
	case syscall.EWINDOWS:
		return false, ": symlinks are not supported on your version of Windows"
	case syscall.ERROR_PRIVILEGE_NOT_HELD:
		return false, ": you don't have enough privileges to create symlinks"
	}

	return false, ""
}
