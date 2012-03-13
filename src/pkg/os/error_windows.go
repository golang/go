// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import "syscall"

// IsExist returns whether the error is known to report that a file already exists.
// It is satisfied by ErrExist as well as some syscall errors.
func IsExist(err error) bool {
	if pe, ok := err.(*PathError); ok {
		err = pe.Err
	}
	return err == syscall.EEXIST || err == syscall.ERROR_ALREADY_EXISTS ||
		err == syscall.ERROR_FILE_EXISTS || err == ErrExist
}

// IsNotExist returns whether the error is known to report that a file does not exist.
// It is satisfied by ErrNotExist as well as some syscall errors.
func IsNotExist(err error) bool {
	if pe, ok := err.(*PathError); ok {
		err = pe.Err
	}
	return err == syscall.ENOENT || err == ErrNotExist
}

// IsPermission returns whether the error is known to report that permission is denied.
// It is satisfied by ErrPermission as well as some syscall errors.
func IsPermission(err error) bool {
	if pe, ok := err.(*PathError); ok {
		err = pe.Err
	}
	return err == syscall.EACCES || err == syscall.EPERM || err == ErrPermission
}
