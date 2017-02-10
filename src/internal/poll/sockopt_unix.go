// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd solaris

package poll

import "syscall"

// SetsockoptByte wraps the setsockopt network call with a byte argument.
func (fd *FD) SetsockoptByte(level, name int, arg byte) error {
	if err := fd.incref(); err != nil {
		return err
	}
	defer fd.decref()
	return syscall.SetsockoptByte(fd.Sysfd, level, name, arg)
}
