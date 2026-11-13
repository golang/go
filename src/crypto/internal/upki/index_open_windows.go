// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package upki

import (
	"internal/syscall/windows"
	"os"
	"path/filepath"
	"syscall"
)

// OpenIndex opens the UPKI index at path. windows.Openat opens the file with
// FILE_SHARE_DELETE, allowing a cache producer to atomically replace the index
// while a check retains this handle.
func OpenIndex(path string) (*os.File, error) {
	parent, err := os.Open(filepath.Dir(path))
	if err != nil {
		return nil, err
	}
	defer parent.Close()

	handle, err := windows.Openat(
		syscall.Handle(parent.Fd()),
		filepath.Base(path),
		syscall.O_RDONLY|syscall.O_CLOEXEC,
		0,
	)
	if err != nil {
		return nil, &os.PathError{Op: "open", Path: path, Err: err}
	}

	return os.NewFile(uintptr(handle), path), nil
}
