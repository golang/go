// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package upki

import (
	"internal/syscall/windows"
	"io"
	"os"
	"path/filepath"
	"syscall"
)

func readFilterFile(path string) ([]byte, error) {
	f, err := openCacheFile(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return io.ReadAll(f)
}

// windows.Openat shares delete access so a cache producer can replace the
// index or remove an obsolete filter while a check is reading it.
func openCacheFile(path string) (*os.File, error) {
	parent, err := os.OpenFile(filepath.Dir(path), os.O_RDONLY|windows.O_DIRECTORY, 0)
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
