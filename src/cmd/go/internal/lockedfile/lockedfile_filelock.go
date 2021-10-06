// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !plan9
// +build !plan9

package lockedfile

import (
	"io/fs"
	"os"

	"cmd/go/internal/lockedfile/internal/filelock"
)

func openFile(name string, flag int, perm fs.FileMode) (*os.File, error) {
	// On BSD systems, we could add the O_SHLOCK or O_EXLOCK flag to the OpenFile
	// call instead of locking separately, but we have to support separate locking
	// calls for Linux and Windows anyway, so it's simpler to use that approach
	// consistently.

	f, err := os.OpenFile(name, flag&^os.O_TRUNC, perm)
	if err != nil {
		return nil, err
	}

	switch flag & (os.O_RDONLY | os.O_WRONLY | os.O_RDWR) {
	case os.O_WRONLY, os.O_RDWR:
		err = filelock.Lock(f)
	default:
		err = filelock.RLock(f)
	}
	if err != nil {
		f.Close()
		return nil, err
	}

	if flag&os.O_TRUNC == os.O_TRUNC {
		if err := f.Truncate(0); err != nil {
			// The documentation for os.O_TRUNC says “if possible, truncate file when
			// opened”, but doesn't define “possible” (golang.org/issue/28699).
			// We'll treat regular files (and symlinks to regular files) as “possible”
			// and ignore errors for the rest.
			if fi, statErr := f.Stat(); statErr != nil || fi.Mode().IsRegular() {
				filelock.Unlock(f)
				f.Close()
				return nil, err
			}
		}
	}

	return f, nil
}

func closeFile(f *os.File) error {
	// Since locking syscalls operate on file descriptors, we must unlock the file
	// while the descriptor is still valid — that is, before the file is closed —
	// and avoid unlocking files that are already closed.
	err := filelock.Unlock(f)

	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	return err
}
