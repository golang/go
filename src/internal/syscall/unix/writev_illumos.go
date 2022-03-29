// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build illumos

package unix

import (
	"syscall"
	"unsafe"
)

//go:cgo_import_dynamic libc_writev writev "libc.so"

//go:linkname procwritev libc_writev

var procwritev uintptr

func Writev(fd int, iovs []syscall.Iovec) (uintptr, error) {
	var p *syscall.Iovec
	if len(iovs) > 0 {
		p = &iovs[0]
	}
	n, _, errno := syscall6(uintptr(unsafe.Pointer(&procwritev)), 3, uintptr(fd), uintptr(unsafe.Pointer(p)), uintptr(len(iovs)), 0, 0, 0)
	if errno != 0 {
		return 0, errno
	}
	return n, nil
}
