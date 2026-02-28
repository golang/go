// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !unix && !windows

package filelock

import (
	"errors"
	"io/fs"
)

type lockType int8

const (
	readLock = iota + 1
	writeLock
)

func lock(f File, lt lockType) error {
	return &fs.PathError{
		Op:   lt.String(),
		Path: f.Name(),
		Err:  errors.ErrUnsupported,
	}
}

func unlock(f File) error {
	return &fs.PathError{
		Op:   "Unlock",
		Path: f.Name(),
		Err:  errors.ErrUnsupported,
	}
}
