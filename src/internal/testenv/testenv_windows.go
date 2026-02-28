// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testenv

import (
	"errors"
	"os"
	"path/filepath"
	"sync"
	"syscall"
)

var hasSymlink = sync.OnceValues(func() (bool, string) {
	tmpdir, err := os.MkdirTemp("", "symtest")
	if err != nil {
		panic("failed to create temp directory: " + err.Error())
	}
	defer os.RemoveAll(tmpdir)

	err = os.Symlink("target", filepath.Join(tmpdir, "symlink"))
	switch {
	case err == nil:
		return true, ""
	case errors.Is(err, syscall.EWINDOWS):
		return false, ": symlinks are not supported on your version of Windows"
	case errors.Is(err, syscall.ERROR_PRIVILEGE_NOT_HELD):
		return false, ": you don't have enough privileges to create symlinks"
	}
	return false, ""
})
