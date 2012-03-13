// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd linux netbsd openbsd

package os

import "syscall"

func isExist(err error) bool {
	if pe, ok := err.(*PathError); ok {
		err = pe.Err
	}
	return err == syscall.EEXIST || err == ErrExist
}

func isNotExist(err error) bool {
	if pe, ok := err.(*PathError); ok {
		err = pe.Err
	}
	return err == syscall.ENOENT || err == ErrNotExist
}

func isPermission(err error) bool {
	if pe, ok := err.(*PathError); ok {
		err = pe.Err
	}
	return err == syscall.EACCES || err == syscall.EPERM || err == ErrPermission
}
