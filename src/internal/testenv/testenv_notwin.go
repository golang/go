// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !windows

package testenv

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sync"
)

var hasSymlink = sync.OnceValues(func() (ok bool, reason string) {
	switch runtime.GOOS {
	case "plan9":
		return false, ""
	case "android", "wasip1":
		// For wasip1, some runtimes forbid absolute symlinks,
		// or symlinks that escape the current working directory.
		// Perform a simple test to see whether the runtime
		// supports symlinks or not. If we get a permission
		// error, the runtime does not support symlinks.
		dir, err := os.MkdirTemp("", "")
		if err != nil {
			return false, ""
		}
		defer func() {
			_ = os.RemoveAll(dir)
		}()
		fpath := filepath.Join(dir, "testfile.txt")
		if err := os.WriteFile(fpath, nil, 0644); err != nil {
			return false, ""
		}
		if err := os.Symlink(fpath, filepath.Join(dir, "testlink")); err != nil {
			if SyscallIsNotSupported(err) {
				return false, fmt.Sprintf("symlinks unsupported: %s", err.Error())
			}
			return false, ""
		}
	}

	return true, ""
})
