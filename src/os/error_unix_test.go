// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build aix || darwin || dragonfly || freebsd || (js && wasm) || linux || netbsd || openbsd || solaris

package os_test

import (
	"io/fs"
	"os"
	"syscall"
)

func init() {
	isExistTests = append(isExistTests,
		isExistTest{err: &fs.PathError{Err: syscall.EEXIST}, is: true, isnot: false},
		isExistTest{err: &fs.PathError{Err: syscall.ENOTEMPTY}, is: true, isnot: false},

		isExistTest{err: &os.LinkError{Err: syscall.EEXIST}, is: true, isnot: false},
		isExistTest{err: &os.LinkError{Err: syscall.ENOTEMPTY}, is: true, isnot: false},

		isExistTest{err: &os.SyscallError{Err: syscall.EEXIST}, is: true, isnot: false},
		isExistTest{err: &os.SyscallError{Err: syscall.ENOTEMPTY}, is: true, isnot: false},
	)
	isPermissionTests = append(isPermissionTests,
		isPermissionTest{err: &fs.PathError{Err: syscall.EACCES}, want: true},
		isPermissionTest{err: &fs.PathError{Err: syscall.EPERM}, want: true},
		isPermissionTest{err: &fs.PathError{Err: syscall.EEXIST}, want: false},

		isPermissionTest{err: &os.LinkError{Err: syscall.EACCES}, want: true},
		isPermissionTest{err: &os.LinkError{Err: syscall.EPERM}, want: true},
		isPermissionTest{err: &os.LinkError{Err: syscall.EEXIST}, want: false},

		isPermissionTest{err: &os.SyscallError{Err: syscall.EACCES}, want: true},
		isPermissionTest{err: &os.SyscallError{Err: syscall.EPERM}, want: true},
		isPermissionTest{err: &os.SyscallError{Err: syscall.EEXIST}, want: false},
	)

}
