// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import "syscall"

func isExist(err error) bool {
	if pe, ok := err.(*PathError); ok {
		err = pe.Err
	}
	return err == syscall.ERROR_ALREADY_EXISTS ||
		err == syscall.ERROR_FILE_EXISTS || err == ErrExist
}

func isNotExist(err error) bool {
	if pe, ok := err.(*PathError); ok {
		err = pe.Err
	}
	return err == syscall.ERROR_FILE_NOT_FOUND ||
		err == syscall.ERROR_PATH_NOT_FOUND || err == ErrNotExist
}

func isPermission(err error) bool {
	if pe, ok := err.(*PathError); ok {
		err = pe.Err
	}
	return err == syscall.ERROR_ACCESS_DENIED || err == ErrPermission
}
