// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd solaris

package unix

import (
	"syscall"
	_ "unsafe" // for go:linkname
)

//go:linkname syscall_fcntl syscall.fcntl
func syscall_fcntl(fd int, cmd int, arg int) (val int, err error)

func IsNonblock(fd int) (nonblocking bool, err error) {
	flag, err := syscall_fcntl(fd, syscall.F_GETFL, 0)
	if err != nil {
		return false, err
	}
	return flag&syscall.O_NONBLOCK != 0, nil
}
