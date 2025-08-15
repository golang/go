// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !windows

package work

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
)

// move moves a file from src to dst setting the permissions
// on the destination file to inherit the permissions from the
// destination parent directory.
func (sh *Shell) move(src, dst string, perm fs.FileMode) error {
	// If the destination directory has the group sticky bit set,
	// we have to copy the file to retain the correct permissions.
	// https://golang.org/issue/18878
	if fi, err := os.Stat(filepath.Dir(dst)); err == nil {
		if fi.IsDir() && (fi.Mode()&fs.ModeSetgid) != 0 {
			return errors.ErrUnsupported
		}
	}
	// The perm argument is meant to be adjusted according to umask,
	// but we don't know what the umask is.
	// Create a dummy file to find out.
	// This works even on systems like Plan 9 where the
	// file mask computation incorporates other information.
	mode := perm
	f, err := os.OpenFile(filepath.Clean(dst)+"-go-tmp-umask", os.O_WRONLY|os.O_CREATE|os.O_EXCL, perm)
	if err == nil {
		fi, err := f.Stat()
		if err == nil {
			mode = fi.Mode() & 0777
		}
		name := f.Name()
		f.Close()
		os.Remove(name)
	}

	if err := os.Chmod(src, mode); err != nil {
		return err
	}
	return os.Rename(src, dst)
}
