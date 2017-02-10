// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux nacl netbsd openbsd solaris windows

package poll

import (
	"io"
	"syscall"
)

// eofError returns io.EOF when fd is available for reading end of
// file.
func (fd *FD) eofError(n int, err error) error {
	if n == 0 && err == nil && fd.ZeroReadIsEOF {
		return io.EOF
	}
	return err
}

// Fchmod wraps syscall.Fchmod.
func (fd *FD) Fchmod(mode uint32) error {
	if err := fd.incref(); err != nil {
		return err
	}
	defer fd.decref()
	return syscall.Fchmod(fd.Sysfd, mode)
}

// Fchown wraps syscall.Fchown.
func (fd *FD) Fchown(uid, gid int) error {
	if err := fd.incref(); err != nil {
		return err
	}
	defer fd.decref()
	return syscall.Fchown(fd.Sysfd, uid, gid)
}

// Ftruncate wraps syscall.Ftruncate.
func (fd *FD) Ftruncate(size int64) error {
	if err := fd.incref(); err != nil {
		return err
	}
	defer fd.decref()
	return syscall.Ftruncate(fd.Sysfd, size)
}

// Fsync wraps syscall.Fsync.
func (fd *FD) Fsync() error {
	if err := fd.incref(); err != nil {
		return err
	}
	defer fd.decref()
	return syscall.Fsync(fd.Sysfd)
}
