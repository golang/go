// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build aix dragonfly freebsd js,wasm linux nacl netbsd openbsd solaris

package poll

import "syscall"

// Fsync wraps syscall.Fsync.
func (fd *FD) Fsync() error {
	if err := fd.incref(); err != nil {
		return err
	}
	defer fd.decref()
	return syscall.Fsync(fd.Sysfd)
}

func fcntl(fd int, cmd int, arg int) (int, error) {
	r, _, e := syscall.Syscall(syscall.SYS_FCNTL, uintptr(fd), uintptr(cmd), uintptr(arg))
	if e != 0 {
		return int(r), syscall.Errno(e)
	}
	return int(r), nil
}
