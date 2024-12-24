// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Supporting definitions for os_uname.go on Solaris.

package osinfo

import (
	"syscall"
	"unsafe"
)

type utsname struct {
	Sysname  [257]byte
	Nodename [257]byte
	Release  [257]byte
	Version  [257]byte
	Machine  [257]byte
}

//go:cgo_import_dynamic libc_uname uname "libc.so"
//go:linkname procUname libc_uname

var procUname uintptr

//go:linkname rawsysvicall6 runtime.syscall_rawsysvicall6
func rawsysvicall6(fn, nargs, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2, err syscall.Errno)

func uname(buf *utsname) error {
	_, _, errno := rawsysvicall6(uintptr(unsafe.Pointer(&procUname)), 1, uintptr(unsafe.Pointer(buf)), 0, 0, 0, 0, 0)
	if errno != 0 {
		return errno
	}
	return nil
}
