// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unix

import (
	"syscall"
	"unsafe"
)

//go:cgo_import_dynamic libc_ioctl ioctl "libc.a/shr_64.o"
//go:linkname libc_ioctl libc_ioctl
var libc_ioctl uintptr

// Implemented in syscall/syscall_aix.go.
func syscall6(trap, nargs, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2 uintptr, err syscall.Errno)

func Ioctl(fd int, cmd int, args unsafe.Pointer) (err error) {
	_, _, e1 := syscall6(uintptr(unsafe.Pointer(&libc_ioctl)), 3, uintptr(fd), uintptr(cmd), uintptr(args), 0, 0, 0)
	if e1 != 0 {
		err = e1
	}
	return
}
