// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build illumos

package unix

import (
	"syscall"
	"unsafe"
)

//go:cgo_import_dynamic libc_pipe2 pipe2 "libc.so"

//go:linkname procpipe2 libc_pipe2

var procpipe2 uintptr

type _C_int int32

func Pipe2(p []int, flags int) error {
	if len(p) != 2 {
		return syscall.EINVAL
	}
	var pp [2]_C_int
	_, _, errno := syscall6(uintptr(unsafe.Pointer(&procpipe2)), 2, uintptr(unsafe.Pointer(&pp)), uintptr(flags), 0, 0, 0, 0)
	if errno != 0 {
		return errno
	}
	p[0] = int(pp[0])
	p[1] = int(pp[1])
	return nil
}
