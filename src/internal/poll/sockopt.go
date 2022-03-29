// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build aix || darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris || windows

package poll

import "syscall"

// SetsockoptInt wraps the setsockopt network call with an int argument.
func (fd *FD) SetsockoptInt(level, name, arg int) error {
	if err := fd.incref(); err != nil {
		return err
	}
	defer fd.decref()
	return syscall.SetsockoptInt(fd.Sysfd, level, name, arg)
}

// SetsockoptInet4Addr wraps the setsockopt network call with an IPv4 address.
func (fd *FD) SetsockoptInet4Addr(level, name int, arg [4]byte) error {
	if err := fd.incref(); err != nil {
		return err
	}
	defer fd.decref()
	return syscall.SetsockoptInet4Addr(fd.Sysfd, level, name, arg)
}

// SetsockoptLinger wraps the setsockopt network call with a Linger argument.
func (fd *FD) SetsockoptLinger(level, name int, l *syscall.Linger) error {
	if err := fd.incref(); err != nil {
		return err
	}
	defer fd.decref()
	return syscall.SetsockoptLinger(fd.Sysfd, level, name, l)
}
