// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build aix dragonfly freebsd linux netbsd openbsd solaris

package unix

import "syscall"

func IsNonblock(fd int) (nonblocking bool, err error) {
	flag, _, e1 := syscall.Syscall(syscall.SYS_FCNTL, uintptr(fd), uintptr(syscall.F_GETFL), 0)
	if e1 != 0 {
		return false, e1
	}
	return flag&syscall.O_NONBLOCK != 0, nil
}
