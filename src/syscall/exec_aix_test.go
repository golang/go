// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build aix

package syscall

import "unsafe"

//go:cgo_import_dynamic libc_Getpgid getpgid "libc.a/shr_64.o"
//go:cgo_import_dynamic libc_Getpgrp getpgrp "libc.a/shr_64.o"

//go:linkname libc_Getpgid libc_Getpgid
//go:linkname libc_Getpgrp libc_Getpgrp

var (
	libc_Getpgid,
	libc_Getpgrp libcFunc
)

func Getpgid(pid int) (pgid int, err error) {
	r0, _, e1 := syscall6(uintptr(unsafe.Pointer(&libc_Getpgid)), 1, uintptr(pid), 0, 0, 0, 0, 0)
	pgid = int(r0)
	if e1 != 0 {
		err = e1
	}
	return
}

func Getpgrp() (pgrp int) {
	r0, _, _ := syscall6(uintptr(unsafe.Pointer(&libc_Getpgrp)), 0, 0, 0, 0, 0, 0, 0)
	pgrp = int(r0)
	return
}

func Tcgetpgrp(fd int) (pgid int32, err error) {
	if errno := ioctlPtr(uintptr(fd), TIOCGPGRP, unsafe.Pointer(&pgid)); errno != 0 {
		return -1, errno
	}
	return pgid, nil
}

func Tcsetpgrp(fd int, pgid int32) (err error) {
	return ioctlPtr(uintptr(fd), TIOCSPGRP, unsafe.Pointer(&pgid))
}
