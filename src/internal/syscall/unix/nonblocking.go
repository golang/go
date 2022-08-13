// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build dragonfly || freebsd || linux || netbsd || openbsd

package unix

import "syscall"

// FcntlSyscall is the number for the fcntl system call. This is
// usually SYS_FCNTL, but can be overridden to SYS_FCNTL64.
var FcntlSyscall uintptr = syscall.SYS_FCNTL

func IsNonblock(fd int) (nonblocking bool, err error) {
	flag, _, e1 := syscall.Syscall(FcntlSyscall, uintptr(fd), uintptr(syscall.F_GETFL), 0)
	if e1 != 0 {
		return false, e1
	}
	return flag&syscall.O_NONBLOCK != 0, nil
}
