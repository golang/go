// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build windows

package os_test

import (
	"os"
	"syscall"
)

func init() {
	const _ERROR_BAD_NETPATH = syscall.Errno(53)

	isExistTests = append(isExistTests,
		isExistTest{err: &os.PathError{Err: syscall.ERROR_FILE_NOT_FOUND}, is: false, isnot: true},
		isExistTest{err: &os.LinkError{Err: syscall.ERROR_FILE_NOT_FOUND}, is: false, isnot: true},
		isExistTest{err: &os.SyscallError{Err: syscall.ERROR_FILE_NOT_FOUND}, is: false, isnot: true},

		isExistTest{err: &os.PathError{Err: _ERROR_BAD_NETPATH}, is: false, isnot: true},
		isExistTest{err: &os.LinkError{Err: _ERROR_BAD_NETPATH}, is: false, isnot: true},
		isExistTest{err: &os.SyscallError{Err: _ERROR_BAD_NETPATH}, is: false, isnot: true},

		isExistTest{err: &os.PathError{Err: syscall.ERROR_PATH_NOT_FOUND}, is: false, isnot: true},
		isExistTest{err: &os.LinkError{Err: syscall.ERROR_PATH_NOT_FOUND}, is: false, isnot: true},
		isExistTest{err: &os.SyscallError{Err: syscall.ERROR_PATH_NOT_FOUND}, is: false, isnot: true},
	)
	isPermissionTests = append(isPermissionTests,
		isPermissionTest{err: &os.PathError{Err: syscall.ERROR_ACCESS_DENIED}, want: true},
		isPermissionTest{err: &os.LinkError{Err: syscall.ERROR_ACCESS_DENIED}, want: true},
		isPermissionTest{err: &os.SyscallError{Err: syscall.ERROR_ACCESS_DENIED}, want: true},
	)
}
