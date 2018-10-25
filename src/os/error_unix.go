// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build aix darwin dragonfly freebsd js,wasm linux nacl netbsd openbsd solaris

package os

import "syscall"

func isExist(err error) bool {
	err = underlyingError(err)
	return err == syscall.EEXIST || err == syscall.ENOTEMPTY || err == ErrExist
}

func isNotExist(err error) bool {
	err = underlyingError(err)
	return err == syscall.ENOENT || err == ErrNotExist
}

func isPermission(err error) bool {
	err = underlyingError(err)
	return err == syscall.EACCES || err == syscall.EPERM || err == ErrPermission
}
