// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build dragonfly freebsd linux netbsd openbsd

package syscall

func Ioctl(fd, req, arg uintptr) (err Errno) {
	_, _, err = Syscall(SYS_IOCTL, fd, req, arg)
	return err
}
