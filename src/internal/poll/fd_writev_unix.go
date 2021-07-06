// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build dragonfly || freebsd || linux || netbsd || openbsd
// +build dragonfly freebsd linux netbsd openbsd

package poll

import (
	"syscall"
	"unsafe"
)

func writev(fd int, iovecs []syscall.Iovec) (uintptr, error) {
	var (
		r uintptr
		e syscall.Errno
	)
	for {
		r, _, e = syscall.Syscall(syscall.SYS_WRITEV, uintptr(fd), uintptr(unsafe.Pointer(&iovecs[0])), uintptr(len(iovecs)))
		if e != syscall.EINTR {
			break
		}
	}
	if e != 0 {
		return r, e
	}
	return r, nil
}
