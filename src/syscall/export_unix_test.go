// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build dragonfly || freebsd || linux || netbsd || openbsd

package syscall

import "unsafe"

func IoctlPtr(fd, req uintptr, arg unsafe.Pointer) (err Errno) {
	_, _, err = Syscall(SYS_IOCTL, fd, req, uintptr(arg))
	return err
}
